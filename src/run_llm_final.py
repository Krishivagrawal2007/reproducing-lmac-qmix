import datetime
import os
from os.path import dirname, abspath
import pprint
import time
import threading
from types import SimpleNamespace as SN
import sys
import importlib
import torch as th
try:
    import wandb
except ImportError:
    wandb = None
import pickle
from controllers import REGISTRY as mac_REGISTRY
from components.episode_buffer import ReplayBuffer
from components.transforms import OneHot
from runners import REGISTRY as r_REGISTRY
from learners import REGISTRY as le_REGISTRY
from modules.meta import REGISTRY as meta_REGISTRY
from utils.general_reward_support import test_alg_config_supports_reward
from utils.logging import Logger
from utils.timehelper import time_left, time_str
from LLM.llm_core import Communication


def run(_run, _config, _log):
    _config = args_sanity_check(_config, _log)
    args = SN(**_config)
    args.device = "cuda" if th.cuda.is_available() else "cpu"
    assert test_alg_config_supports_reward(args), \
        "The specified algorithm does not support the general reward setup. Please choose a different algorithm or set `common_reward=True`."

    phase = getattr(args, "phase", None)
    comm_code_paths = getattr(args, "comm_code_paths", None)
    important_state = getattr(args, "important_state", None)

    logger = Logger(_log)
    _log.info("Experiment Parameters:")
    experiment_params = pprint.pformat(_config, indent=4, width=1)
    _log.info("\n\n" + experiment_params + "\n")
    
    
    if _config["env"] == 'lbf':
        players = _config["env_args"]["players"]
        max_player_level = _config["env_args"]["max_player_level"]
        field_size = _config["env_args"]["field_size"]
        max_food = _config["env_args"]["max_food"]
        sight = _config["env_args"]["sight"]
        force_coop = "-coop" if _config["env_args"]["force_coop"] else ""
        _config["env_args"]["key"] = "Foraging-{}x{}-{}p-{}f-s{}{}".format(
            str(field_size),
            str(field_size),
            str(players),
            str(max_food),
            str(sight),
            str(force_coop)
        )

    
    try:
        map_name = _config["env_args"]["map_name"]
    except:
        map_name = _config["env_args"]["key"]
    unique_token = (
        f"{_config['name']}_seed{_config['seed']}_{map_name}_{datetime.datetime.now()}"
                    )
    args.unique_token = unique_token

    if args.use_tensorboard:
        tb_logs_direc = os.path.join(dirname(dirname(abspath(__file__))), "results", "tb_logs")
        tb_exp_direc = os.path.join(tb_logs_direc, "{}").format(unique_token)
        logger.setup_tb(tb_exp_direc)
    if args.use_wandb:
        logger.setup_wandb(
            _config, args.wandb_team, args.wandb_project, args.wandb_mode
        )
    
    logger.setup_sacred(_run)

    print("use_wandb:", args.use_wandb, args.wandb_project, args.wandb_team, args.wandb_mode)

    run_sequential(args=args, logger=logger,
                   comm_code_paths=comm_code_paths,
                   saved_important_state=important_state)
    logger.finish()
    print("Finish RL")
    
    # Clean up after finishing
    print("Exiting Main")
    print("Stopping all threads")
    for t in threading.enumerate():
        if t.name != "MainThread":
            print("Thread {} is alive! Is daemon: {}".format(t.name, t.daemon))
            t.join(timeout=1)
            print("Thread joined")
    print("Exiting script")
    
    return

def evaluate_sequential(args, runner):
    for _ in range(args.test_nepisode):
        runner.run(test_mode=True)

    if args.save_replay:
        runner.save_replay()

    runner.close_env()

def run_sequential(args, logger, comm_code_paths=None, saved_important_state=None):
    
    runner = r_REGISTRY[args.runner](args=args, logger=logger)
    env_info = runner.get_env_info()
    args.n_agents = env_info["n_agents"]
    args.n_actions = env_info["n_actions"]
    args.state_shape = env_info["state_shape"]
    # args.n_enemies = env_info["n_enemies"]
    if "unit_dim" in env_info:
        args.unit_dim = env_info["unit_dim"]

    scheme = {
        "state": {"vshape": env_info["state_shape"]},
        "obs": {"vshape": env_info["obs_shape"], "group": "agents"},
        "actions": {"vshape": (1,), "group": "agents", "dtype": th.long},
        "avail_actions": {"vshape": (env_info["n_actions"],), "group": "agents", "dtype": th.int},
        "terminated": {"vshape": (1,), "dtype": th.uint8},
        "goals": {"vshape": (1,), "group": "agents", "dtype": th.long},
        "flag_win": {"vshape": (1,), "dtype": th.uint8},
        "reward": {"vshape": (1,)} if args.common_reward else {"vshape": (args.n_agents,)},
    }
    
    groups = {"agents": args.n_agents}
    preprocess = {"actions": ("actions_onehot", [OneHot(out_dim=args.n_actions)])}

    buffer = ReplayBuffer(
        args, scheme, groups, args.buffer_size, env_info["episode_limit"] + 1,
        preprocess=preprocess,
        device="cpu" if args.buffer_cpu_only else args.device,
    )

    comm_code_paths = comm_code_paths or getattr(args, "comm_code_paths", [])
    if not comm_code_paths:
        arg_paths = getattr(args, "comm_code_path", None)
        comm_code_paths.append(arg_paths)
        
    important_state = saved_important_state or getattr(args, "important_state", None)

    if comm_code_paths and important_state is not None:
        comm_modules = []
        message_dims = []
        if not hasattr(args, "phase") or args.phase is None:
            args.phase = "multi_train"
        comm = Communication(args, None)
        comm_module = comm.code_utils.import_and_reload_module(comm_code_paths[0])

        test_obs = th.randn(args.batch_size, args.n_agents, env_info['obs_shape'] + args.n_agents + args.n_actions)
        message_dim = comm_module.communication(test_obs).shape[-1] - (env_info['obs_shape'] + args.n_agents + args.n_actions)
        comm_modules.append(comm_module)
        message_dims.append(message_dim)

        if comm_code_paths.__len__() >= 2:
            comm_module_add = comm.code_utils.import_and_reload_module(comm_code_paths[1])
            message_dim_add = comm_module_add.communication(test_obs).shape[-1] - (env_info['obs_shape'] + args.n_agents + args.n_actions)
            comm_modules.append(comm_module_add)
            message_dims.append(message_dim_add)

        for i in range(2, len(comm_code_paths)):
            test_obs = th.randn(args.batch_size, 10, args.n_agents, env_info['obs_shape'] + args.n_agents + args.n_actions)
            comm_module_updated_timewise = comm.code_utils.import_and_reload_module(comm_code_paths[i])
            message_dim_updated_timewise = comm_module_updated_timewise.communication(test_obs).shape[-1] - (env_info['obs_shape'] + args.n_agents + args.n_actions)
            comm_modules.append(comm_module_updated_timewise)
            message_dims.append(message_dim_updated_timewise)

        if args.use_wandb:
            log_important_info_to_wandb(comm_code_paths[0], important_state, label="initial")

            if  comm_code_paths.__len__() >= 2:
                log_important_info_to_wandb(comm_code_paths[1], important_state, label="add")

            for i in range(2, len(comm_code_paths)):
                log_important_info_to_wandb(comm_code_paths[i], important_state, label="timewise")

    else:
        raise RuntimeError("comm_code_path and important_state must be specified for multi_train phase.")


    mac = mac_REGISTRY[args.mac](buffer.scheme, groups, comm_modules, message_dims, important_state, args)
    runner.setup(scheme=scheme, groups=groups, preprocess=preprocess, mac=mac)
    learner = le_REGISTRY[args.learner](mac, buffer.scheme, important_state, logger, args)
    if args.use_cuda:
        learner.cuda()

    if getattr(args, "checkpoint_path", ""):
        if os.path.isdir(args.checkpoint_path):
            timesteps = [int(name) for name in os.listdir(args.checkpoint_path) if name.isdigit()]
            if timesteps:
                timestep_to_load = max(timesteps) if args.load_step == 0 else min(timesteps, key=lambda x: abs(x - args.load_step))
                model_path = os.path.join(args.checkpoint_path, str(timestep_to_load))
                logger.console_logger.info(f"Loading model from {model_path}")
                learner.load_models(model_path)
                runner.t_env = timestep_to_load
                if args.evaluate or args.save_replay:
                    runner.log_train_stats_t = runner.t_env
                    evaluate_sequential(args, runner)
                    logger.log_stat("episode", runner.t_env, runner.t_env)
                    logger.print_recent_stats()
                    logger.console_logger.info("Finished Evaluation")
                    return
        else:
            logger.console_logger.error(f"Checkpoint directory {args.checkpoint_path} doesn't exist")

    
    episode = 0
    last_test_T = -args.test_interval - 1
    last_log_T = 0
    model_save_time = 0
    start_time = time.time()
    last_time = start_time
    logger.console_logger.info(f"Beginning training for {args.t_max} timesteps")

    if args.debug_mode:
        if args.save_batch:
            while not buffer.can_sample(args.batch_size):
                with th.no_grad():
                    episode_batch = runner.run(test_mode=True)
                    buffer.insert_episode_batch(episode_batch)
            episode_sample = buffer.sample(args.batch_size)
            with open('episode_sample.pkl', 'wb') as file:
                pickle.dump(episode_sample, file)
            logger.console_logger.info(f"Saved episode sample as episode_sample.pkl")
            exit(0)
        else:
            with open('episode_sample.pkl', 'rb') as file:
                episode_sample = pickle.load(file)
            episode_sample = episode_sample[:, :episode_sample.max_t_filled()]
            if episode_sample.device != args.device:
                episode_sample.to(args.device)

            logger.console_logger.info(f"Loaded episode sample")
            learner.train(episode_sample, runner.t_env, episode)
            exit(0)

    while runner.t_env <= args.t_max:
        episode_batch = runner.run(test_mode=False)
        
        buffer.insert_episode_batch(episode_batch)
        if buffer.can_sample(args.batch_size):
            episode_sample = buffer.sample(args.batch_size)
            max_ep_t = episode_sample.max_t_filled()
            episode_sample = episode_sample[:, :max_ep_t]
            if episode_sample.device != args.device:
                episode_sample.to(args.device)
            learner.train(episode_sample, runner.t_env, episode)

        n_test_runs = max(1, args.test_nepisode // runner.batch_size)
        if (runner.t_env - last_test_T) / args.test_interval >= 1.0:
            logger.console_logger.info(f"t_env: {runner.t_env} / {args.t_max}")
            logger.console_logger.info(
                f"Estimated time left: {time_left(last_time, last_test_T, runner.t_env, args.t_max)}. "
                f"Time passed: {time_str(time.time() - start_time)}"
            )
            last_time = time.time()
            last_test_T = runner.t_env
            for _ in range(n_test_runs):
                runner.run(test_mode=True)

        if args.save_model and (
            runner.t_env - model_save_time >= args.save_model_interval or model_save_time == 0
        ):
            model_save_time = runner.t_env
            save_path = os.path.join(args.local_results_path, "models", args.unique_token, str(runner.t_env))
            os.makedirs(save_path, exist_ok=True)
            logger.console_logger.info(f"Saving models to {save_path}")
            learner.save_models(save_path)

        episode += args.batch_size_run
        if (runner.t_env - last_log_T) >= args.log_interval:
            logger.log_stat("episode", episode, runner.t_env)
            logger.print_recent_stats()
            last_log_T = runner.t_env

    runner.close_env()
    logger.console_logger.info("Finished Training")

    done_flag_path = os.path.join(args.save_dir, "done.flag")
    with open(done_flag_path, "w") as f:
        f.write("done\n")
    logger.console_logger.info(f"[DONE] Written: {done_flag_path}")

def args_sanity_check(config, _log):
    if config["use_cuda"] and not th.cuda.is_available():
        config["use_cuda"] = False
        _log.warning(
            "CUDA flag use_cuda was switched OFF automatically because no CUDA devices are available!"
        )

    if config["test_nepisode"] < config["batch_size_run"]:
        config["test_nepisode"] = config["batch_size_run"]
    else:
        config["test_nepisode"] = (
            config["test_nepisode"] // config["batch_size_run"]
        ) * config["batch_size_run"]

    return config

def import_and_reload_module(module_name):
    if module_name in sys.modules:
        del sys.modules[module_name]
    return importlib.import_module(module_name)


def phase0_comm(args):
    test_obs = th.randn(args.batch_size, args.n_agents, args.obs_shape + args.n_agents + args.n_actions)
    comm = Communication(args, test_obs)

    imp_code_path, important_state = comm.imp_state_generate(max_retries=args.max_retries)
    comm_code_path, comm_module, message_dim = comm.init_comm_generate(max_retries=args.max_retries, important_dims=important_state)
    
    print("[Phase0] comm_code_path:", comm_code_path)
    print("[Phase0] important_state:", important_state)
    return comm_code_path, important_state

def phase1_comm_update_basic(comm_code_path, important_state, feedback_data_path, args):

    print(f"[Phase1] Loading phase0 discriminator results for baseline comparison...")
    
    phase0_comm_file = os.path.join(feedback_data_path, "feedback_results_phase0_initial.pkl")
    baseline_file = os.path.join(feedback_data_path, "feedback_results_baseline.pkl")
    
    if os.path.exists(phase0_comm_file):
        print(f"[Phase1] Loading Phase1 communication results from: {phase0_comm_file}")
        with open(phase0_comm_file, "rb") as f:
            phase1_data = pickle.load(f)
        comm_feedback_data = phase1_data['squared_error']
    else:
        raise FileNotFoundError(f"Phase1 communication results not found: {phase0_comm_file}")
    
    baseline_feedback_data = None
    if os.path.exists(baseline_file):
        print(f"[Phase1] Loading baseline results from: {baseline_file}")
        with open(baseline_file, "rb") as f:
            baseline_data = pickle.load(f)
        baseline_feedback_data = baseline_data['squared_error']
    else:
        print(f"[Phase1] Warning: Baseline results not found: {baseline_file}")
        print(f"[Phase1] Proceeding without baseline comparison")

    from utils.logging import Logger
    from runners import REGISTRY as r_REGISTRY
    logger = Logger(None)
    runner = r_REGISTRY[args.runner](args=args, logger=logger)
    env_info = runner.get_env_info()
    args.n_agents = env_info["n_agents"]
    args.n_actions = env_info["n_actions"]
    args.state_shape = env_info["state_shape"]

    test_obs = th.randn(args.batch_size, args.n_agents, env_info['obs_shape'] + args.n_agents + args.n_actions)
    comm = Communication(args, test_obs)

    with open(comm_code_path, "r") as f:
        cur_comm_code = f.read()

    feedback, feedback_path = comm.feedback_generate(
        feedback_data=comm_feedback_data, 
        imp_state=important_state, 
        threshold=args.mse_thres,
        cur_communication_method=cur_comm_code, 
        timestep_wise=False, 
        max_retries=args.max_retries,
        baseline_feedback_data=baseline_feedback_data
    )
    update_code_path, update_code_module, _ = comm.comm_update(
        feedback=feedback,
        cur_communication_method=cur_comm_code,
        timestep_wise=False,
        max_retries=args.max_retries
    )
    
    comparison_str = "with baseline comparison" if baseline_feedback_data is not None else "without baseline comparison"
    print(f"[Phase2] Basic communication update completed {comparison_str}: {update_code_path}")
    return update_code_path, feedback_path

def timestepwise_comm_update(phase, comm_code_path, important_state, feedback_data_path, args):
    
    print(f"[Phase{phase}] Loading phase{phase-1} discriminator results for timestep-wise analysis...")

    if phase==2:
        prev_phase_comm_file = os.path.join(feedback_data_path, "feedback_results_phase1_updated.pkl")
    else:
        prev_phase_comm_file = os.path.join(feedback_data_path, f"feedback_results_phase{phase-1}.pkl")
    baseline_file = os.path.join(feedback_data_path, "feedback_results_baseline.pkl")
    
    if os.path.exists(prev_phase_comm_file):
        print(f"[Phase{phase}] Loading Phase{phase-1} communication results from: {prev_phase_comm_file}")
        with open(prev_phase_comm_file, "rb") as f:
            prev_phase_data = pickle.load(f)
        comm_feedback_data = prev_phase_data['squared_error']
    else:
        raise FileNotFoundError(f"Phase{phase} communication results not found: {prev_phase_comm_file}")
    
    baseline_feedback_data = None
    if os.path.exists(baseline_file):
        print(f"[Phase{phase}] Loading baseline results from: {baseline_file}")
        with open(baseline_file, "rb") as f:
            baseline_data = pickle.load(f)
        baseline_feedback_data = baseline_data['squared_error']
    else:
        print(f"[Phase{phase}] Warning: Baseline results not found: {baseline_file}")
        print(f"[Phase{phase}] Proceeding without baseline comparison")

    from utils.logging import Logger
    from runners import REGISTRY as r_REGISTRY
    logger = Logger(None)
    runner = r_REGISTRY[args.runner](args=args, logger=logger)
    env_info = runner.get_env_info()
    args.n_agents = env_info["n_agents"]
    args.n_actions = env_info["n_actions"]
    args.state_shape = env_info["state_shape"]

    test_obs = th.randn(args.batch_size, args.n_agents, env_info['obs_shape'] + args.n_agents + args.n_actions)
    comm = Communication(args, test_obs)

    with open(comm_code_path, "r") as f:
        cur_comm_code = f.read()

    feedback, feedback_path = comm.feedback_generate(
        feedback_data=comm_feedback_data, 
        imp_state=important_state, 
        threshold=args.mse_thres,
        cur_communication_method=cur_comm_code, 
        timestep_wise=True, 
        max_retries=args.max_retries,
        baseline_feedback_data=baseline_feedback_data
    )
    update_code_path_timewise, update_code_module_timewise, _ = comm.comm_update(
        feedback=feedback,
        cur_communication_method=cur_comm_code,
        timestep_wise=True,
        max_retries=args.max_retries,
        phase=phase
    )
    
    comparison_str = "with baseline comparison" if baseline_feedback_data is not None else "without baseline comparison"
    print(f"[Phase{phase}] Timestep-wise communication update completed {comparison_str}: {update_code_path_timewise}")
    return update_code_path_timewise, feedback_path


def log_important_info_to_wandb(comm_code_path, important_state, label="main"):
    try:
        with open(comm_code_path, "r") as f:
            comm_code = f.read()

        comm_code_html = wandb.Html(f"<h4>Comm Module Code ({label})</h4><pre>{comm_code}</pre>")
        important_state_html = wandb.Html(f"<h4>Important States ({label})</h4><pre>{important_state}</pre>")

        wandb.log({
            f"comm_module_code_{label}": comm_code_html,
            f"important_state_{label}": important_state_html
        }, step=0)

    except Exception as e:
        print(f"Failed to log comm module info for {label}: {e}")