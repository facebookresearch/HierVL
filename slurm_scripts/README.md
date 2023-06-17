## Running on SLURM cluster

We use SLURM cluster for all our experiments. This folder contains helper code for running the code on SLURM.

### Copying and running the code

Submitted SLURM scripts run as per the priority. In order to run modify the code between submitting the script and running it, we copy the code to a different directory to make sure we can edit the code in the original location. Thus, we first copy the code to a new location and then submit it from there.

**NOTE**: We need to put absolute path for datasets that we do not copy. We only copy scripts that are typically a few MBs in size.

### Submitting to SLURM

To submit a job, simply run

```bash
bash mover_trainer.sh job_name
```

