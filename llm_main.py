import os

import pandas as pd
from datetime import datetime
from src.llm.utils import load_prompt_config, parse_args, add_index_as_column
from src.llm.runner import annotate_df


if __name__ == "__main__":
    args = parse_args()

    # Base dirs (kept stable for tmux / server usage)
    data_folder_dir = "/Users/tianxiaozhang/Desktop/py_playground/suicide_study/data"
    yaml_folder_dir = "/Users/tianxiaozhang/Desktop/py_playground/suicide_study/prompts"

    data_path = os.path.join(data_folder_dir, args.data_name)
    yaml_path = os.path.join(yaml_folder_dir, args.yaml_name)

    # Load inputs
    rsd15k = pd.read_csv(data_path)
    print(f"Data loaded from : {data_path}")
    yaml_settings = load_prompt_config(yaml_path)
    print(f"Running {yaml_settings['theory_name']} thoery with LLM {yaml_settings['llm_name']}")

    sample_size = yaml_settings.get("sample_size")
    if sample_size is not None:
        print(f"Sampling {sample_size} rows (seed={yaml_settings['random_seed']})")
        running_df = rsd15k.sample(
            n=sample_size,
            random_state=yaml_settings["random_seed"],
        ).reset_index(drop=True)
    else:
        running_df = rsd15k.copy()
        sample_size = len(rsd15k)
        print(f"Running with the full dataset with {sample_size} rows")

    df_annotated = annotate_df(running_df, yaml_settings, text_col="text")

    # Output
    output_dir = (
        f"{yaml_settings['output_path']}/"
        f"{datetime.now().strftime('%Y%m%d')}_"
        f"{yaml_settings['theory_name']}_"
        f"{yaml_settings['llm_name']}"
    )
    output_path = os.path.join(
        output_dir,
        f"rsd15k_feature_{yaml_settings['theory_name']}_{sample_size}_{yaml_settings['llm_name']}.csv",
    )
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    df_annotated.to_csv(output_path, index=False)
    print(f"Done. Output saved to: {output_path}")

# example cmd
# python llm_main.py --data_name rsd1000rand.csv --yaml_name psycho_concepts1.yml

