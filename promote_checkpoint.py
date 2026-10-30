"""
Promote a checkpoint (from checkpoints/, or any other .zip) into
swing_policy.zip + best_model/ in one step, instead of typing out several
cp commands by hand -- that manual sequence is exactly the kind of thing
that's easy to fumble (wrong path, forgetting a file, forgetting to back
up first), and this collapses it into one command with the risky parts
handled automatically.

WHAT IT DOES, IN ORDER:
  1. Backs up whatever's currently in swing_policy.zip, vec_normalize.pkl,
     and best_model/ into a fresh timestamped folder -- nothing is ever
     just overwritten without a copy kept.
  2. Copies the chosen checkpoint into BOTH swing_policy.zip (what
     train.py resumes from) and best_model/best_model.zip (what
     evaluate.py checks by default) -- both need updating for a promotion
     to actually take effect; copying only one was a mistake made earlier
     in this project.
  3. Copies the current vec_normalize.pkl into
     best_model/best_vec_normalize.pkl to keep them paired (see the
     caveat below).
  4. Archives evaluate_log.csv (if present) to a timestamped file and
     starts a fresh one. WHY: auto_train.py's decline/stall/streak checks
     read consecutive best_model/best_model rows as if they're one
     continuous trend. A promotion deliberately swaps in different
     weights -- comparing a post-promotion round against pre-promotion
     history isn't a fair "is training declining" signal, it's comparing
     across a deliberate reset. Starting a fresh log means the next
     several rounds are judged only against each other, not against
     whatever happened before this promotion.

CAVEAT, same as scan_checkpoints.py: checkpoints from checkpoints/ don't
have their own matched VecNormalize snapshot unless they were saved after
the train.py save_vecnormalize fix -- this uses whichever vec_normalize.pkl
currently exists, an approximation for anything but the very latest
checkpoint.

Run locally:
    python promote_checkpoint.py checkpoints/swing_ppo_4863232_steps.zip
"""
import os
import shutil
import sys
from datetime import datetime, timezone

SWING_POLICY = "swing_policy.zip"
VECNORM_PATH = "vec_normalize.pkl"
BEST_MODEL_DIR = "best_model"
BEST_MODEL_ZIP = os.path.join(BEST_MODEL_DIR, "best_model.zip")
BEST_VECNORM = os.path.join(BEST_MODEL_DIR, "best_vec_normalize.pkl")
LOG_PATH = "evaluate_log.csv"


def main():
    if len(sys.argv) != 2:
        print("Usage: python promote_checkpoint.py <path-to-checkpoint.zip>")
        print("Example: python promote_checkpoint.py checkpoints/swing_ppo_4863232_steps.zip")
        sys.exit(1)

    checkpoint_path = sys.argv[1]
    if not os.path.exists(checkpoint_path):
        print(f"!! {checkpoint_path} doesn't exist -- nothing to promote.")
        sys.exit(1)
    if not os.path.exists(VECNORM_PATH):
        print(f"!! {VECNORM_PATH} not found -- need it to pair with the promoted checkpoint.")
        sys.exit(1)

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")

    # 1. Back up current state.
    backup_dir = f"pre_promotion_backup_{stamp}"
    os.makedirs(backup_dir, exist_ok=True)
    for src in [SWING_POLICY, VECNORM_PATH, BEST_MODEL_ZIP, BEST_VECNORM]:
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(backup_dir, os.path.basename(src)))
    print(f"Backed up current state to {backup_dir}/")

    # 2 & 3. Promote.
    os.makedirs(BEST_MODEL_DIR, exist_ok=True)
    shutil.copy2(checkpoint_path, SWING_POLICY)
    shutil.copy2(checkpoint_path, BEST_MODEL_ZIP)
    shutil.copy2(VECNORM_PATH, BEST_VECNORM)
    print(f"Promoted {checkpoint_path} -> {SWING_POLICY} and {BEST_MODEL_ZIP}")
    print(f"Paired it with current {VECNORM_PATH} -> {BEST_VECNORM}")

    # 4. Fresh evaluate_log.csv so trend detection doesn't span the
    # promotion boundary.
    if os.path.exists(LOG_PATH):
        archive_path = f"evaluate_log_before_promotion_{stamp}.csv"
        os.rename(LOG_PATH, archive_path)
        print(f"Archived old {LOG_PATH} -> {archive_path} (nothing lost, still "
              f"readable) and starting a fresh log, so auto_train.py's decline/"
              f"stall/streak checks judge the next rounds against each other, "
              f"not against pre-promotion history.")
    else:
        print(f"No existing {LOG_PATH} to archive -- starting fresh anyway.")

    print(f"\nDone. Ready to resume with: python auto_train.py")
    print("Reminder: if you were in the middle of verifying a candidate with "
          "evaluate.py, put its MODEL_PATH/VECNORM_PATH back to "
          "\"best_model/best_model\" / \"vec_normalize.pkl\" before running "
          "auto_train.py, since it runs evaluate.py as-is.")


if __name__ == "__main__":
    main()
