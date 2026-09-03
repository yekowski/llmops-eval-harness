import os

def write_to_step_summary(markdown_content: str):
    """Writes evaluation report to the GITHUB_STEP_SUMMARY file when running in GitHub Actions."""
    summary_file = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary_file:
        try:
            with open(summary_file, "a") as f:
                f.write("\n" + markdown_content + "\n")
            print("Report written to GITHUB_STEP_SUMMARY.")
        except Exception as e:
            print(f"Failed to write to GITHUB_STEP_SUMMARY: {e}")
    else:
        print("GITHUB_STEP_SUMMARY env var not set. Skipping step summary output.")
