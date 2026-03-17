import json
import os
import sys
import hydra
import wandb
import warnings
from tqdm import tqdm
from omegaconf import OmegaConf
from hydra.core.config_store import ConfigStore
from dotenv import load_dotenv

from src.classroom import Classroom, JudgeDecision
from utils.data import load_datasets
from config.eval import EvalConfig
from src.utils.utils import init_logger

load_dotenv()
logger = init_logger()
cs = ConfigStore.instance()
cs.store(name="config", node=EvalConfig)
warnings.filterwarnings("ignore")


def cleanup_classroom(classroom):
    for attr in ["teacher_model", "student_model", "judge_model", "reward_model"]:
        model = getattr(classroom, attr, None)
        if model is None:
            continue
        cleanup = getattr(model, "cleanup", None)
        if callable(cleanup):
            try:
                cleanup()
            except Exception as exc:
                logger.warning(f"Failed to cleanup {attr}: {exc}")


@hydra.main(config_path="config/eval", version_base=None)
def main(cfg: EvalConfig):
    # Merge loaded config with defaults
    default_config = OmegaConf.structured(EvalConfig)
    cfg = OmegaConf.merge(default_config, cfg)

    # Initialize wandb logging if enabled in the config
    if hasattr(cfg, "logging") and cfg.logging.get("wandb", False):
        wandb.init(
            project=cfg.logging.wandb_project,
            name=cfg.logging.wandb_run_name,
            entity=cfg.logging.wandb_entity,
            group=cfg.logging.run_group,
            tags=cfg.logging.wandb_tags,
            config=OmegaConf.to_object(cfg),
        )
        logger.info("Initialized wandb logging.")

    logger.info("Loading evaluation data and constructing the Classroom instance...")

    # Instantiate the Classroom for evaluation
    classroom = Classroom(
        cfg.student_model,
        cfg.teacher_model,
        cfg.judge_model,
        cfg.reward_model,
        cfg.generation,
        None,
    )

    # Load evaluation datasets
    _, eval_data = load_datasets(cfg.dataset, cfg.seed)
    print(eval_data)

    _problems_we_sample = eval_data["problem"]
    _answers_we_sample = eval_data["answer"]

    number_of_times_to_average = cfg.num_samples_per_problem

    problem_we_sample = []
    answer_we_sample = []
    for i in range(len(_problems_we_sample)):
        problem_we_sample.extend([_problems_we_sample[i]] * number_of_times_to_average)
        answer_we_sample.extend([_answers_we_sample[i]] * number_of_times_to_average)

    logger.info("Sampling conversations...")
    conversations = classroom.sample_conversations(
        problem_we_sample,
        answer_we_sample,
        compute_initial_attempt=cfg.recompute_initial_attempts,
    )

    logger.info("Computing metrics...")
    metrics = {}

    if cfg.recompute_initial_attempts:
        # Compute reward deltas across conversations
        deltas = []
        for i in tqdm(
            range(len(_problems_we_sample)),
            desc="Metric aggregation: delta/initial",
            leave=True,
        ):
            current_deltas = []
            for j in range(number_of_times_to_average):
                current_deltas.append(
                    conversations[
                        i * number_of_times_to_average + j
                    ].get_end_rm_reward()
                    - conversations[
                        i * number_of_times_to_average + j
                    ].get_initial_rm_reward()
                )
            deltas.append(sum(current_deltas) / len(current_deltas))
        delta_mean = sum(deltas) / len(deltas)
        print(f"Delta mean: {delta_mean}")
        metrics["delta_mean"] = delta_mean

        # Mean before
        initial_rm_rewards = []
        for i in range(len(_problems_we_sample)):
            current_rewards = []
            for j in range(number_of_times_to_average):
                current_rewards.append(
                    conversations[
                        i * number_of_times_to_average + j
                    ].get_initial_rm_reward()
                )
            initial_rm_rewards.append(sum(current_rewards) / len(current_rewards))
        initial_rm_mean = sum(initial_rm_rewards) / len(initial_rm_rewards)
        print(f"Initial RM mean: {initial_rm_mean}")
        metrics["initial_rm_rewards_mean"] = initial_rm_mean

    # Mean after
    end_rm_rewards = []
    for i in tqdm(
        range(len(_problems_we_sample)),
        desc="Metric aggregation: end reward",
        leave=True,
    ):
        current_rewards = []
        for j in range(number_of_times_to_average):
            current_rewards.append(
                conversations[i * number_of_times_to_average + j].get_end_rm_reward()
            )
        end_rm_rewards.append(sum(current_rewards) / len(current_rewards))
    end_rm_mean = sum(end_rm_rewards) / len(end_rm_rewards)
    print(f"End RM mean: {end_rm_mean}")
    metrics["end_rm_rewards_mean"] = end_rm_mean

    # Compute leaked solutions rate across conversations
    leaked_solutions = []
    for i in tqdm(
        range(len(_problems_we_sample)),
        desc="Metric aggregation: leakage",
        leave=True,
    ):
        current_decisions = []
        for j in range(number_of_times_to_average):
            decisions = [
                d.decision
                for d in conversations[
                    i * number_of_times_to_average + j
                ].judge_decisions["does_not_leak_answer"]
            ]
            current_decisions.append(
                decisions.count(JudgeDecision.REJECT) / len(decisions)
            )
        leaked_solutions.append(sum(current_decisions) / len(current_decisions))

    leaked_mean = sum(leaked_solutions) / len(leaked_solutions)
    print(f"Leaked solutions mean: {leaked_mean}")
    metrics["leaked_solutions_mean"] = leaked_mean

    # Compute the rate at which pedagogical values are rejected
    follows_pedagogical_values = []
    for i in tqdm(
        range(len(_problems_we_sample)),
        desc="Metric aggregation: pedagogy",
        leave=True,
    ):
        current_decisions = []
        for j in range(number_of_times_to_average):
            decisions = [
                d.decision
                for d in conversations[
                    i * number_of_times_to_average + j
                ].judge_decisions["follows_pedagogical_values"]
            ]
            current_decisions.append(
                decisions.count(JudgeDecision.REJECT) / len(decisions)
            )
        follows_pedagogical_values.append(
            sum(current_decisions) / len(current_decisions)
        )

    does_not_follow_mean = sum(follows_pedagogical_values) / len(
        follows_pedagogical_values
    )
    print(f"Does not follow pedagogical values mean: {does_not_follow_mean}")
    metrics["rejects_pedagogical_values_mean"] = does_not_follow_mean

    df_table = classroom.to_pd_latest()

    if cfg.score_using_pedagogical_reward:
        from utils.pedagogical_reward import score_each_conversation

        scores = score_each_conversation(df_table, cfg.pedagogical_reward_model)

        # Scores is a list of lists
        df_table["pedagogical_reward"] = scores
        # We compute the mean pedagogical reward
        pedagogical_rewards = [
            sum([float(s) for s in score]) / len(score) if len(score) != 0 else 0
            for score in scores
        ]
        pedagogical_reward_macro_avg = sum(pedagogical_rewards) / len(
            pedagogical_rewards
        )
        print(f"Pedagogical reward mean macro avg: {pedagogical_reward_macro_avg}")
        metrics["pedagogical_reward_macro_avg"] = pedagogical_reward_macro_avg

        # Micro average
        pedagogical_reward_micro_avg = sum(
            [float(s) for score in scores for s in score]
        ) / sum([len(score) for score in scores])
        print(f"Pedagogical reward mean micro avg: {pedagogical_reward_micro_avg}")
        metrics["pedagogical_reward_micro_avg"] = pedagogical_reward_micro_avg

    rewards = [classroom.get_end_rm_reward(c) for c in conversations]
    df_table["end_rm_reward"] = rewards
    rewards = [classroom.get_thinking_reward(c) for c in conversations]
    df_table["thinking_reward"] = rewards
    rewards = [classroom.get_end_of_conversation_reward(c) for c in conversations]
    df_table["end_of_conversation_reward"] = rewards
    rewards = [classroom.get_length_reward(c) for c in conversations]
    df_table["length_reward"] = rewards

    df_table["total_reward"] = (
        df_table["end_rm_reward"]
        + df_table["thinking_reward"]
        + df_table["end_of_conversation_reward"]
        + df_table["length_reward"]
    )

    if cfg.export_conversations_path:
        export_path = os.path.abspath(cfg.export_conversations_path)
        os.makedirs(os.path.dirname(export_path), exist_ok=True)
        if export_path.endswith(".json"):
            df_table.to_json(export_path, orient="records", force_ascii=False, indent=2)
        elif export_path.endswith(".csv"):
            df_table.to_csv(export_path, index=False)
        else:
            raise ValueError("export_conversations_path must end with .json or .csv")
        logger.info(f"Saved conversation table to {export_path}")

    if cfg.export_metrics_path:
        export_metrics_path = os.path.abspath(cfg.export_metrics_path)
        os.makedirs(os.path.dirname(export_metrics_path), exist_ok=True)
        with open(export_metrics_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)
        logger.info(f"Saved metrics to {export_metrics_path}")

    # Log metrics to wandb if enabled
    if hasattr(cfg, "logging") and cfg.logging.get("wandb", False):
        wandb.log(
            {
                "delta_mean": metrics.get("delta_mean", 0),
                "initial_rm_rewards_mean": metrics.get("initial_rm_rewards_mean", 0),
                "end_rm_rewards_mean": end_rm_mean,
                "leaked_solutions_mean": leaked_mean,
                "rejects_pedagogical_values_mean": does_not_follow_mean,
                "pedagogical_reward_macro_avg": metrics.get(
                    "pedagogical_reward_macro_avg", 0
                ),
                "pedagogical_reward_micro_avg": metrics.get(
                    "pedagogical_reward_micro_avg", 0
                ),
            }
        )
        df_table = df_table.astype(str)
        print(df_table)
        if cfg.logging.wandb:
            wandb.log({"conversations": wandb.Table(dataframe=df_table)})
        wandb.finish()

    cleanup_classroom(classroom)
    sys.stdout.flush()
    sys.stderr.flush()
    raise SystemExit(0)


if __name__ == "__main__":
    main()
