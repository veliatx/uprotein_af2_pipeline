from datetime import datetime
import os
import boto3
import json
import math
from collections import defaultdict
import multiprocessing
import pandas as pd


class AlphaFold_Multimer:
    """
    Class for alphafold2 peptide receptor prediction.
    """

    def __init__(self, data_dir, receptor, receptor_start, receptor_end, peptides_dict_file, num_ensemble,
                 random_seed, models_to_use, time_stamp, use_spot_instance=False, 
                 root_path = '/home/ubuntu/efs/mnt/efs0/alphafold', filter_completed = True, benchmark=False):
        self.job_name = f'{receptor}_{time_stamp}'
        self.docker_image = '328315166908.dkr.ecr.us-west-2.amazonaws.com/alphafold:latest'
        self.job_definition = 'arn:aws:batch:us-west-2:328315166908:job-definition/alphafold:11'
        if use_spot_instance:
            self.job_queue = 'bfx-jq-gpu-spot'
        else:
            self.job_queue = 'bfx-jq-gpu'
        self.data_dir = data_dir
        self.root_path = root_path

        if filter_completed:
            with open(os.path.join(root_path, 'peptides', peptides_dict_file), 'r') as f:
                peptides = json.load(f)
            filtered_peptide_dict = self._filter_completed_results(receptor, peptides, [i+'_pred_0' for i in models_to_use.split(',')], seed = random_seed)
            if len(filtered_peptide_dict) > 0:
                self.peptides_dict_file = f"{self.job_name}.peptides.filtered.json"
                with open(os.path.join(f"{root_path}/peptides/", self.peptides_dict_file), 'w') as f:
                    json.dump(filtered_peptide_dict, f)
            else:
                self.peptides_dict_file = ''
        else:
            self.peptides_dict_file = peptides_dict_file

        self.command = [str(c) for c in 
            [
                'python', 'run_alphafold_multimer_velia.py',
                '--data_dir', data_dir,
                '--receptor', receptor,
                '--receptor_start', receptor_start,
                '--num_ensemble', num_ensemble,
                '--random_seed', random_seed,
                '--models_to_use', models_to_use,
                '--run_multimer_system',]
            ]

        if receptor_end > 0:
            self.command += ['--receptor_end', str(receptor_end)]
        if benchmark:
            self.command += ['--benchmark']
        self.command_test = ['nvidia-smi']


    def submit(self, n_experiments_limit=0, n_experiments_per_instance=-1):
        if self.peptides_dict_file == '':
            print('Skipping because no unfinished results exist.')
            return
        
        if n_experiments_per_instance == -1:
            self._submit_a_job(
                    batch_index=-1,
                    peptides_dict_file=self.peptides_dict_file,
                    n_experiments_limit=n_experiments_limit)
        else:
            batched_peptides_filenames = self._batch_jobs(n_experiments_per_instance)
            for batch_index in range(len(batched_peptides_filenames)):
                self._submit_a_job(
                        batch_index=batch_index,
                        peptides_dict_file=batched_peptides_filenames[batch_index],
                        n_experiments_limit=n_experiments_limit)


    def local_run(self):
        docker_run_cmd = (
            "docker run --rm  -u 1000 --gpus all "
            f"-v /home/ubuntu/efs/mnt/efs0/alphafold/:{self.data_dir} alphafold ")
        command = self.command + [
            '--n_experiments_limit', 0,
            '--peptides_dict_file', self.peptides_dict_file],
        docker_run_cmd += " ".join(command)
        os.system(docker_run_cmd)


    def _submit_a_job(self, batch_index, n_experiments_limit, peptides_dict_file):
        response = boto3.client('batch', 'us-west-2').submit_job(
            jobName = self.job_name+f"_{batch_index}",
            jobQueue = self.job_queue,
            jobDefinition = self.job_definition,
            containerOverrides = {
                'command': self.command + [
                    '--n_experiments_limit', str(n_experiments_limit),
                    '--peptides_dict_file', peptides_dict_file],
            }
        )


    def _batch_jobs(self, n_experiments_per_instance):
        peptide_dict = json.loads(
            open(f"{self.root_path}/peptides/{self.peptides_dict_file}"
            ).read())
        
        jobs = []
        for peptide_name, coordinates in peptide_dict.items():
            for coordinate in coordinates:
                jobs.append((peptide_name, coordinate))

        batched_peptides_filenames = []
        for batch_index in range(math.ceil(len(jobs) / float(n_experiments_per_instance))):
            current_peptides_dict = {}
            for k, v in jobs[batch_index*n_experiments_per_instance:(batch_index+1)*n_experiments_per_instance]:
                if k not in current_peptides_dict:
                    current_peptides_dict[k] = [v]
                else:
                    current_peptides_dict[k].append(v)

            current_peptide_pathname = f"{self.job_name}_batch_{batch_index}_{self.peptides_dict_file}"
            current_peptide_path = f"{self.root_path}/peptides/{current_peptide_pathname}"
            batched_peptides_filenames.append(current_peptide_pathname)
            with open(current_peptide_path, "w") as ofile:
                json.dump(current_peptides_dict, ofile)
        
        return batched_peptides_filenames
    

    # Function to check if 'predict_output.pkl' exists in a given prefix
    def _check_file_exists(self, bucket, prefix):
        objects = list(bucket.objects.filter(Prefix=prefix))
        if len(objects)>0:
            return True
        else:
            return False

    def _check_result_exists(self, receptor, peptide, model, seed = 13982528712754415):
        bucket_name = 'velia-af2-dev'
        s3_resource = boto3.resource('s3')
        bucket = s3_resource.Bucket(bucket_name)
        result_name = f"{receptor}_{peptide}_{seed}"
        file_prefix = f"outputs/{result_name}/{result_name}.{model}.done.txt"
        # print(file_prefix)
        exists = self._check_file_exists(bucket, file_prefix)
        return exists

    def _filter_completed_results(self, receptor, peptides, models, seed = 13982528712754415, parallel = True):
        peptides_to_run = defaultdict(list)
        inputs = []
        for peptide_name, intervals in peptides.items():
            for start, stop in intervals:
                for current_model in models:
                    if parallel:
                        inputs.append((receptor, f"{peptide_name}_{start}_{stop}", current_model, seed))
                        done = False
                    else:
                        done = self._check_result_exists(receptor, f"{peptide_name}_{start}_{stop}", current_model, seed)
                        if done:
                            pass
                        else:
                            if (start, stop) not in peptides_to_run[peptide_name]:
                                peptides_to_run[peptide_name].append((start, stop))
        if parallel:
            with multiprocessing.Pool(32) as ppool:
                results = ppool.starmap(self._check_result_exists, inputs)
            for r, i in zip(results, inputs):
                if not r:
                    _, pept_name, _, _ = i
                    pep = pept_name.split('_')
                    start = int(pep[-2])
                    stop = int(pep[-1])
                    if (start, stop) not in peptides_to_run['_'.join(pep[:-2])]:
                        peptides_to_run['_'.join(pep[:-2])].append((start, stop))
        return peptides_to_run


if __name__ == '__main__':

    local = False
    time_stamp = datetime.today().strftime("%Y%m%d%H%M%S")

    if local:
        data_dir = "/app/alphafold/alphafold_data"
        for receptor in ['GPR18_HUMAN', 'MC3R_HUMAN', 'MC4R_HUMAN', 'NK1R_HUMAN', 'NK2R_HUMAN', 'GIPR_HUMAN']:
            job = AlphaFold_Multimer(
                data_dir=data_dir,
                receptor=receptor,
                peptides_dict_file="peptides_062024.json",
                num_ensemble=1,
                random_seed=13982528712754415,
                models_to_use="model_1_multimer_v3,model_2_multimer_v3,model_3_multimer_v3,model_4_multimer_v3,model_5_multimer_v3",
                time_stamp=time_stamp)
            job.local_run()
            break
    else:
        data_dir = "/mount/efs/alphafold/"
        local_root_path = '/home/ubuntu/efs/mnt/efs0/alphafold'
        #receptor_file = "positive_control_receptors_simplified.tsv"
        #receptor_file = "uniprot_singlePass_transmembrane_receptors_simplified.tsv"
        #receptor_file = "uniprot_GPCR_receptors_simplified.tsv"
        receptor_file = "de_receptors_membrane_list.tsv"
        receptor_df = pd.read_table(
                receptor_file, index_col=0, sep=','
            ).astype({"receptor_start": int, "receptor_end": int})
        for row in receptor_df.iterrows():
            pth_to_features = f"{local_root_path}/outputs/monomer_alignments/{row[1]['Entry Name']}/monomer_features.pkl"
            if not os.path.exists(pth_to_features):
                print(row[1]["Entry Name"])
                continue

            job = AlphaFold_Multimer(
                data_dir=data_dir,
                receptor=row[1]["Entry Name"],
                receptor_start=row[1]["receptor_start"],
                receptor_end=row[1]["receptor_end"],
                peptides_dict_file="FDCSP.json",
                num_ensemble=1,
                models_to_use="model_1_multimer_v3",
                random_seed=13982528712754420,
                time_stamp=time_stamp,
                use_spot_instance=True,
                root_path=local_root_path,
                benchmark=False)
            #job.submit(n_experiments_limit=6, n_experiments_per_instance=6)
            job.submit(n_experiments_limit=1, n_experiments_per_instance=50)
            # models_to_use="model_1_multimer_v3",
            # models_to_use="model_1_multimer_v3,model_2_multimer_v3,model_3_multimer_v3,model_4_multimer_v3,model_5_multimer_v3",

