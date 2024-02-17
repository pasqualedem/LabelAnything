import os
from st_aggrid import AgGrid

import streamlit as st
import pandas as pd
import subprocess

OUT_DIR = 'out'

@st.cache_data
def get_slurm_jobs():
        # Run sacct command and capture output
        command_output = subprocess.run(['sacct', '--format=JobID,JobName,State,Start,End,Elapsed,CPUTime,AllocCPUS,Partition'],
                                        capture_output=True, text=True)

        # Check if the command was successful
        if command_output.returncode == 0:
            # Split the output into lines and create a list of dictionaries
            output_lines = command_output.stdout.strip().split('\n')
            data_list = [line.split() for line in output_lines[2:]]  # Skip header line
            columns = output_lines[0].split()

            # Create a DataFrame from the list of dictionaries
            df = pd.DataFrame(data_list, columns=columns)
            return df
        else:
            df = pd.DataFrame([])
            

def show_slurm():
    if st.button('Refresh'):
        st.cache_data.clear()
    with st.expander('Slurm'):
        df = get_slurm_jobs()
        AgGrid(df)
    
            
def parse_miou(output):
    mious = []
    with open(output) as f:
                    lines = f.readlines()
                    for line in lines:
                        if 'miou:' in line:
                            mious.append(float(line.split('miou:')[1].strip()))
    return mious


def show_run(group_dir, run):
    run_config = os.path.join(group_dir, f'{run}.yaml')
    run_output = os.path.join(group_dir, f'{run}.out')
    with st.expander('Show config'):
        if os.path.exists(run_config):
            with open(run_config) as f:
                st.code(f.read())
        else:
            st.write(f'No config file found: {run_config}')
    if os.path.exists(run_output):
        # Check if "wandb sync has been written to the output file"
        with open(run_output) as f:
            lines = f.readlines()
            found = False
            for line in lines:
                if 'wandb sync' in line:
                    command = line.split('wandb sync')[1]
                    command += f"> {group_dir}/{run}.sync 2>&1 &"
                    st.code(command)
                    found = True
                    break
        if not found:
            st.write('No wandb sync found')
        if st.button("Show miou"):
            miou = parse_miou(run_output)
            if len(miou) > 0:
                st.line_chart(pd.DataFrame(miou, columns=['miou']))
    else:
        st.write(f'No output file found: {run_output}')
    if os.path.exists(os.path.join(group_dir, f'{run}.sync')):
        with st.expander('Show sync'):
            with open(os.path.join(group_dir, f'{run}.sync')) as f:
                st.code(f.read())
        
def get_run_data(group_dir, run):
    outfile = os.path.join(group_dir, f'{run}.out')
    if os.path.exists(outfile):
        unix_time = os.path.getmtime(outfile)
        return {"Out File": outfile, "Last Modified": pd.to_datetime(unix_time, unit='s')}
    else:
        return None, None

def show_group(group):
    group_dir = os.path.join(OUT_DIR, group)
    # Get last modified time
    runs = os.listdir(group_dir)
    runs = set([run.split('.')[0] for run in runs])
    runs_data = pd.DataFrame([{"Run": run, **get_run_data(group_dir, run)} for run in runs])
    AgGrid(runs_data)   
    
    run = st.selectbox('Run', list(runs))
    show_run(group_dir, run)
    
        
def show_groups():
    groups = sorted(os.listdir(OUT_DIR), reverse=True)
    if "old" in groups:
        groups.remove("old")
    group = st.selectbox('Group', groups)
    show_group(group)
            

def main():
    st.title('Experiment handler')
    show_slurm()
    show_groups()

if __name__ == '__main__':
    main()
    