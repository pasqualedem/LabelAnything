import os
from st_aggrid import AgGrid

import streamlit as st
import pandas as pd
import subprocess

OUT_DIR = "out"


@st.cache_resource
def get_slurm_jobs():
    def highlight_state(val):
        if val == "RUNNING":
            return "background-color: blue"
        elif val == "FAILED":
            return "background-color: red"
        elif val == "COMPLETED":
            return "background-color: green"
        elif val == "PENDING":
            return "background-color: yellow"
        else:
            return ""

    # Run sacct command and capture output
    command_output = subprocess.run(
        [
            "sacct",
            "--format=JobID,JobName,State,Start,End,Elapsed,CPUTime,AllocCPUS,Partition",
        ],
        capture_output=True,
        text=True,
    )

    # Check if the command was successful
    if command_output.returncode == 0:
        # Split the output into lines and create a list of dictionaries
        output_lines = command_output.stdout.strip().split("\n")
        data_list = [line.split() for line in output_lines[2:]]  # Skip header line
        columns = output_lines[0].split()

        # Create a DataFrame from the list of dictionaries
        df = pd.DataFrame(data_list, columns=columns)
        df = df[df["JobID"].str.isnumeric()]
        # Set running jobs background to yellow, failed to red, and completed to green
        styled_df = df.style.applymap(highlight_state, subset=["State"])
        return styled_df
    else:
        df = pd.DataFrame([])


def show_slurm():
    if st.button("Refresh"):
        st.cache_resource.clear()
    with st.expander("Slurm"):
        df = get_slurm_jobs()
        st.dataframe(df, use_container_width=True)


def parse_miou(output):
    mious = []
    with open(output) as f:
        lines = f.readlines()
        for line in lines:
            if "miou:" in line:
                mious.append(float(line.split("miou:")[1].strip()))
    return mious


def read_output(output, lines=10):
    if lines < 0:
        slc = slice(lines, None)
    else:
        slc = slice(None, lines)
    with open(output) as f:
        return "\n".join(f.readlines()[slc])


def show_run(group_dir, run):
    run_config = os.path.join(group_dir, f"{run}.yaml")
    run_output = os.path.join(group_dir, f"{run}.out")
    with st.expander("Show config"):
        if os.path.exists(run_config):
            with open(run_config) as f:
                st.code(f.read())
        else:
            st.write(f"No config file found: {run_config}")
    if os.path.exists(run_output):
        # Check if "wandb sync has been written to the output file"
        with open(run_output) as f:
            lines = f.readlines()
            found = False
            for line in lines:
                if "wandb sync" in line:
                    folder = line.split("wandb sync")[1].strip()
                    command = f"wandb sync {folder} > {group_dir}/{run}.sync 2>&1 &"
                    st.code(command, language="bash")
                    found = True
                    break
        if not found:
            st.write("No wandb sync found")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            miou_button = st.button("Show miou")
        with col3:
            lines = st.number_input("Lines", min_value=-100, max_value=100, value=-10)
        with col4:
            out_button = st.button("Show output")
        if out_button:
            st.code(read_output(run_output, lines=lines))
        if miou_button:
            miou = parse_miou(run_output)
            if len(miou) > 0:
                st.line_chart(pd.DataFrame(miou, columns=["miou"]))
    else:
        st.write(f"No output file found: {run_output}")
    if os.path.exists(os.path.join(group_dir, f"{run}.sync")):
        with st.expander("Show sync"):
            with open(os.path.join(group_dir, f"{run}.sync")) as f:
                st.code(f.read())


def get_run_data(group_dir, run):
    outfile = os.path.join(group_dir, f"{run}.out")
    if os.path.exists(outfile):
        unix_time = os.path.getmtime(outfile)
        return {
            "Out File": outfile,
            "Last Modified": pd.to_datetime(unix_time, unit="s"),
        }
    else:
        return {"Out File": None, "Last Modified": None}


def show_group(group):
    group_dir = os.path.join(OUT_DIR, group)
    # Get last modified time
    runs = os.listdir(group_dir)
    runs = set([run.split(".")[0] for run in runs])
    runs_data = pd.DataFrame(
        [{"Run": run, **get_run_data(group_dir, run)} for run in runs]
    )
    st.dataframe(runs_data, use_container_width=True)

    run = st.selectbox("Run", list(runs))
    show_run(group_dir, run)


def show_groups():
    groups = sorted(os.listdir(OUT_DIR), reverse=True)
    if "old" in groups:
        groups.remove("old")
    group = st.selectbox("Group", groups)
    show_group(group)


def launch_job():
    sh_file = "launch_experiment_exe"
    slurm_file = st.text_input("Slurm file", value="launch_run")
    exe_file = st.text_input("Exe file", value="launch_run_exe")
    col1, col2 = st.columns(2)
    with col1:
        if os.path.exists(slurm_file):
            with open(slurm_file) as f:
                st.code(f.read())
        else:
            st.write(f"No slurm file found: {slurm_file}")
    with col2:
        if os.path.exists(exe_file):
            with open(exe_file) as f:
                st.code(f.read())
        else:
            st.write(f"No exe file found: {exe_file}")
    params = [p for p in os.listdir() if p.endswith(".yaml")]
    if os.path.exists("parameters"):
        params += [
            os.path.join("parameters", param_file)
            for param_file in os.listdir("parameters")
        ]
    param = st.selectbox("Param", params)
    only_create = st.checkbox("Only create")
    command = (
        f'sh ./{sh_file} --parameters={param} {"--only-create" if only_create else ""}'
    )
    st.code(command, language="bash")
    command = command.split()
    if st.button("Run"):
        with st.spinner("Running"):
            # Get output
            output = subprocess.run(command, capture_output=True, text=True)
        st.code(output.stdout)
        st.code(output.stderr)
        st.balloons()


def main():
    st.title("Experiment handler")
    viewer_tab, launcher_tab = st.tabs((f"Viewer", f"Launcher"))
    with viewer_tab:
        show_slurm()
        show_groups()
    with launcher_tab:
        launch_job()


if __name__ == "__main__":
    main()
