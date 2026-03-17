from codecarbon import track_emissions, EmissionsTracker, OfflineEmissionsTracker
from codecarbon.core.util import backup
from codecarbon.external.logger import logger
from codecarbon.output_methods.base_output import BaseOutput
from codecarbon.output_methods.emissions_data import EmissionsData, TaskEmissionsData
from typing import List
import codecarbon
import codecarbon.core.cpu 
import logging
import os
import csv
import pandas as pd
import src.config as config
import src.trainer.stats.base as base
import torch
import threading
import time
import uuid

logger = logging.getLogger(__name__)

# artificially force psutil to fail, so that CodeCarbon uses constant mode for CPU measurements
codecarbon.core.cpu.is_psutil_available = lambda: False

trainer_stats_name="codecarbon2"

def construct_trainer_stats(conf : config.Config, **kwargs) -> base.TrainerStats:
    if "device" in kwargs:
        device = kwargs["device"]
    else:
        logger.warning("No device provided to codecarbon trainer stats. Using default PyTorch device")
        device = torch.get_default_device() 
    return CodeCarbonStats(device, conf.trainer_stats_configs.codecarbon.run_num, conf.trainer_stats_configs.codecarbon.project_name, conf.trainer_stats_configs.codecarbon.output_dir)

class SimpleFileOutput(BaseOutput): 
    
    def __init__(self, 
        output_file_name: str = "codecarbon.csv", 
        output_dir: str = ".",
        on_csv_write: str = "append"
    ):
        if on_csv_write not in {"append", "update"}:
            raise ValueError(
                f"Unknown `on_csv_write` value: {on_csv_write}"
                + " (should be one of 'append' or 'update'"
            )
        
        self.output_file_name: str = output_file_name
        
        if not os.path.exists(output_dir):
            raise OSError(f"Folder '{output_dir}' doesn't exist !")
        
        self.output_dir: str = output_dir
        self.on_csv_write: str = on_csv_write
        self.save_file_path = os.path.join(self.output_dir, self.output_file_name) #default: ./codecarbon.csv
        
        logger.info(
        f"Emissions data (if any) will be saved to file {os.path.abspath(self.save_file_path)}"
        )

    def has_valid_headers(self, data: EmissionsData):
        with open(self.save_file_path) as csv_file:
            csv_reader = csv.DictReader(csv_file)
            dict_from_csv = dict(list(csv_reader)[0])
            list_of_column_names = list(dict_from_csv.keys())
            return list(data.values.keys()) == list_of_column_names

    def to_csv(self, total: EmissionsData, delta: EmissionsData):
        """
        Save the emissions data to a CSV file.
        If the file already exists, append the new data to it.
        param `delta` is not used in this method.
        """

        # Add code to check whether a part of the save_file_path already exists -->
        # in our case, the output_file_name-experiment_name file exists, but the current code
        # is only checking whether output_file_name exists. Since it doesn't, it goes ahead 
        # and creates this new file, despite it being a "useless" file.

        # Problem: stop() calls persist_data() which calls out() and task_out(). Thus the task csv file is accurate.
        # But is out() accurate with tasks? How do tasks update total_emissions and delta? 

        file_exists: bool = os.path.isfile(self.save_file_path)
        
        if file_exists and not self.has_valid_headers(total): # CSV headers changed
            logger.warning("The CSV format have changed, backing up old emission file.")
            backup(self.save_file_path)
            file_exists = False 
        
        new_df = pd.DataFrame.from_records([dict(total.values)])

        if not file_exists:
            df = new_df
        elif self.on_csv_write == "append":
            df = pd.read_csv(self.save_file_path)
            df = pd.concat([df, new_df])
        else:
            df = pd.read_csv(self.save_file_path)
            df_run = df.loc[df.run_id == total.run_id]
            if len(df_run) < 1:
                df = pd.concat([df, new_df])
            elif len(df_run) > 1:
                logger.warning(
                f"CSV contains more than 1 ({len(df_run)})"
                + f" rows with current run ID ({total.run_id})."
                + "Appending instead of updating."
                )
                df = pd.concat([df, new_df])
            else:
                df.at[df.run_id == total.run_id, total.values.keys()] = total.values.values()
    
        df.to_csv(self.save_file_path, index=False)

    def out(self, total: EmissionsData, delta: EmissionsData):
        self.to_csv(total, delta)

    def live_out(self, total: EmissionsData, delta: EmissionsData):
        pass

    def task_out(self, data: List[TaskEmissionsData], experiment_name: str):
        # run_id = data[0].run_id
        split = os.path.splitext(self.save_file_path)
        save_task_file_path = split[0] + "-" + experiment_name + split[1]
        df = pd.DataFrame(columns=data[0].values.keys())
        new_df = pd.DataFrame.from_records(
            [dict(data_point.values) for data_point in data]
        )
        # Filter out empty or all-NA columns, to avoid warnings from Pandas
        new_df = new_df.dropna(axis=1, how="all")
        df = pd.concat([df, new_df], ignore_index=True)
        df.to_csv(save_task_file_path, index=False)

class CodeCarbonStats(base.TrainerStats):
    """Provides energy consumed and carbon emitted during model training.

    This version includes a background sampler that records CodeCarbon tasks
    every `sample_interval_ms` milliseconds (default 500ms). Checkpoint periods
    also emit explicit tasks via start_save_checkpoint / stop_save_checkpoint.
    """

    def __init__(
        self,
        device : torch.device,
        run_num : int,
        project_name : str,
        output_dir : str,
        sample_interval_ms: int = 500,
    ) -> None:
        # Track current iteration number in the training loop
        self.iteration = 0

        # CUDA device indicates the current GPU assigned to this process (0, 1, 2, ...)
        self.device = device
        self.run_num = run_num
        run_number = f"run_{run_num}_"
        gpu_id = self.device.index if getattr(self.device, "index", None) is not None else 0

        # log the losses
        self.losses = []
        self.project_name = project_name
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

        # Save sampling interval in seconds
        self.sample_interval_s = float(sample_interval_ms) / 1000.0

        # Normal-mode tracker to track the entire training loop
        self.total_training_tracker = OfflineEmissionsTracker(
            project_name = project_name,
            country_iso_code = "CAN",
            region = "quebec",
            save_to_file = False,
            output_handlers = [SimpleFileOutput(output_file_name = f"{run_number}cc_full_rank_{gpu_id}.csv", output_dir=output_dir)],
            allow_multiple_runs = True,
            log_level = "warning",
            gpu_ids = [gpu_id],
        )

        # Task-mode tracker to track steps (iterations) within the training loop
        self.training_step_tracker = OfflineEmissionsTracker(
            project_name = project_name,
            experiment_name = "steps",
            country_iso_code = "CAN",
            region = "quebec",
            save_to_file = False,
            output_handlers = [SimpleFileOutput(output_file_name = f"{run_number}cc_step_rank_{gpu_id}.csv", output_dir=output_dir)],
            allow_multiple_runs = True,
            api_call_interval = -1,
            gpu_ids = [gpu_id],
            log_level = "warning",
        )

        # Task-mode tracker to track individual substeps (forward/backward/opt)
        self.training_substep_tracker = OfflineEmissionsTracker(
            project_name = project_name,
            experiment_name = "substeps",
            country_iso_code = "CAN",
            region = "quebec",
            save_to_file = False,
            output_handlers = [SimpleFileOutput(output_file_name = f"{run_number}cc_substep_rank_{gpu_id}.csv", output_dir=output_dir)],
            allow_multiple_runs = True,
            api_call_interval = -1,
            gpu_ids = [gpu_id],
            log_level = "warning",
        )

        # --- Sampling tracker: records short tasks every sample_interval_s ---
        self.sampling_tracker = OfflineEmissionsTracker(
            project_name = project_name,
            experiment_name = "samples",
            country_iso_code = "CAN",
            region = "quebec",
            save_to_file = False,
            output_handlers = [SimpleFileOutput(output_file_name = f"{run_number}cc_samples_rank_{gpu_id}.csv", output_dir=output_dir)],
            allow_multiple_runs = True,
            api_call_interval = -1,
            gpu_ids = [gpu_id],
            log_level = "warning",
        )

        # initialise task-mode trackers
        self.training_substep_tracker.start()
        self.training_step_tracker.start()

        # sampling thread control
        self._sampling_stop_event = threading.Event()
        self._sampling_thread = threading.Thread(target=self._sampling_loop, name=f"cc-sampler-run{run_num}-gpu{gpu_id}", daemon=True)
        self._sampling_thread_started = False

    def _sampling_loop(self):
        """Background loop: start_task -> wait interval -> stop_task repeatedly."""
        sample_idx = 0
        while not self._sampling_stop_event.is_set():
            task_name = f"sample_{sample_idx}"
            try:
                # start a short task that will last ~self.sample_interval_s
                self.sampling_tracker.start_task(task_name=task_name)
            except Exception as e:
                logger.warning(f"CodeCarbon sampling start_task failed: {e}")

            # wait for the interval, but allow early exit
            stopped_early = self._sampling_stop_event.wait(self.sample_interval_s)

            try:
                self.sampling_tracker.stop_task(task_name=task_name)
            except Exception as e:
                logger.warning(f"CodeCarbon sampling stop_task failed: {e}")

            sample_idx += 1
            if stopped_early:
                break

    def start_train(self) -> None:
        torch.cuda.synchronize(self.device)
        # start the total tracker
        self.total_training_tracker.start()
        # start the sampling tracker and its thread
        try:
            self.sampling_tracker.start()
        except Exception as e:
            logger.warning(f"Failed to start sampling_tracker: {e}")
        self._sampling_stop_event.clear()
        if not self._sampling_thread.is_alive():
            self._sampling_thread = threading.Thread(target=self._sampling_loop, name=self._sampling_thread.name, daemon=True)
            self._sampling_thread.start()
            self._sampling_thread_started = True

    def stop_train(self) -> None:
        torch.cuda.synchronize(self.device)
        # stop the total tracker
        try:
            self.total_training_tracker.stop()
        except Exception as e:
            logger.warning(f"Failed to stop total_training_tracker: {e}")

        # signal sampling thread to stop, join, then stop sampling tracker
        self._sampling_stop_event.set()
        if self._sampling_thread_started:
            self._sampling_thread.join(timeout=5.0)  # give it up to 5s to finish
            self._sampling_thread_started = False
        try:
            self.sampling_tracker.stop()
        except Exception as e:
            logger.warning(f"Failed to stop sampling_tracker: {e}")

        # stop other trackers (step/substep)
        try:
            self.training_step_tracker.stop()
        except Exception as e:
            logger.warning(f"Failed to stop training_step_tracker: {e}")
        try:
            self.training_substep_tracker.stop()
        except Exception as e:
            logger.warning(f"Failed to stop training_substep_tracker: {e}")

    def start_step(self) -> None:
        self.iteration += 1
        torch.cuda.synchronize(self.device)
        self.training_step_tracker.start_task(task_name = f"Step #{self.iteration}")

    def stop_step(self) -> None:
        torch.cuda.synchronize(self.device)
        self.training_step_tracker.stop_task(task_name = f"Step #{self.iteration}")

    def start_forward(self) -> None:
        torch.cuda.synchronize(self.device)
        self.training_substep_tracker.start_task(task_name = f"Forward pass #{self.iteration}")

    def stop_forward(self) -> None:
        torch.cuda.synchronize(self.device)
        self.training_substep_tracker.stop_task(task_name = f"Forward pass #{self.iteration}")

    def start_backward(self) -> None:
        torch.cuda.synchronize(self.device)
        self.training_substep_tracker.start_task(task_name = f"Backward pass #{self.iteration}")

    def stop_backward(self) -> None:
        torch.cuda.synchronize(self.device)
        self.training_substep_tracker.stop_task(task_name = f"Backward pass #{self.iteration}")

    def start_optimizer_step(self) -> None:
        torch.cuda.synchronize(self.device)
        self.training_substep_tracker.start_task(task_name = f"Optimisation step #{self.iteration}")

    def stop_optimizer_step(self) -> None:
        torch.cuda.synchronize(self.device)
        self.training_substep_tracker.stop_task(task_name = f"Optimisation step #{self.iteration}")

    # --- explicit checkpoint tasks so they appear in the samples CSV as named events ---
    def start_save_checkpoint(self) -> None:
        torch.cuda.synchronize(self.device)
        try:
            self.sampling_tracker.start_task(task_name = f"checkpoint_iter_{self.iteration}")
        except Exception as e:
            logger.warning(f"Failed to start checkpoint sampling task: {e}")

    def stop_save_checkpoint(self) -> None:
        torch.cuda.synchronize(self.device)
        try:
            self.sampling_tracker.stop_task(task_name = f"checkpoint_iter_{self.iteration}")
        except Exception as e:
            logger.warning(f"Failed to stop checkpoint sampling task: {e}")

    def log_step(self) -> None:
        pass

    def log_stats(self) -> None:
        """
        Log the loss statistics to an external file.
        """
        # losses as dataframe
        df = pd.DataFrame([[x["task_name"], x["loss"].item()] for x in self.losses])

        # save to file ({output_dir}/losses/run_{run_num}_cc_loss_rank_{gpu_id}.csv)
        run_number = f"run_{self.run_num}_"
        gpu_id = self.device.index if getattr(self.device, "index", None) is not None else 0
        losses_dir = os.path.join(self.output_dir, "losses")
        os.makedirs(losses_dir, exist_ok=True)
        save_file_path = os.path.join(losses_dir, f"{run_number}cc_loss_rank_{gpu_id}.csv")
        df.to_csv(save_file_path, index=False)

        logger.info(f"CODECARBON LOSS LOGGING: Rank {gpu_id} - Run {self.run_num} - Losses saved to {save_file_path}")

    def log_loss(self, loss: torch.Tensor) -> None:
        """
        Take the loss from the training loop and log it to the CodeCarbon tracker file.
        """
        self.losses.append(
            {
                "task_name": f"Step #{self.iteration}",
                "loss": loss.to(torch.device("cpu"), non_blocking=True),
            }
        )
