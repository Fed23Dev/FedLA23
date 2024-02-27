import numpy as np
import torch.cuda

from env.running_env import args, global_logger
from env.support_config import VState
from federal.federal_util import simulation_federal_process, get_data_ratio
from federal.simulation.Master import FedAvgMaster, FedProxMaster, FedLAMaster, ScaffoldMaster, MoonMaster
from utils.MathTools import js_divergence


def test_fedavg():
    loader, loaders, _ = simulation_federal_process()
    master_node = FedAvgMaster(workers=args.workers, activists=args.active_workers, local_epoch=args.local_epoch,
                               loader=loader, workers_loaders=loaders)
    master_node.union_run(args.federal_round)
    master_node.cell.exit_proc(one_key=f'{args.exp_name}-test_acc')


def test_fedprox():
    loader, loaders, _ = simulation_federal_process()
    master_node = FedProxMaster(workers=args.workers, activists=args.active_workers, local_epoch=args.local_epoch,
                                loader=loader, workers_loaders=loaders)
    master_node.union_run(args.federal_round)
    master_node.cell.exit_proc(one_key=f'{args.exp_name}-test_acc')


def test_fedla():
    loader, loaders, user_dict = simulation_federal_process()
    # global_dist, device_ratios = get_data_ratio(user_dict)

    master_node = FedLAMaster(workers=args.workers, activists=args.active_workers, local_epoch=args.local_epoch,
                              loader=loader, workers_loaders=loaders, num_classes=args.num_classes,
                              clusters=args.clusters, drag=args.drag, threshold=args.threshold)
    master_node.union_run(args.federal_round)
    master_node.cell.exit_proc(one_key=f'{args.exp_name}-test_acc')


def test_scaffold():
    loader, loaders, user_dict = simulation_federal_process()
    master_node = ScaffoldMaster(workers=args.workers, activists=args.active_workers, local_epoch=args.local_epoch,
                                 loader=loader, workers_loaders=loaders, local_batch=args.batch_limit)
    master_node.union_run(args.federal_round)
    master_node.cell.exit_proc(one_key=f'{args.exp_name}-test_acc')


def test_moon():
    loader, loaders, user_dict = simulation_federal_process()
    master_node = MoonMaster(workers=args.workers, activists=args.active_workers, local_epoch=args.local_epoch,
                             loader=loader, workers_loaders=loaders, mu=args.mu, T=args.T)
    master_node.union_run(args.federal_round)
    master_node.cell.exit_proc(one_key=f'{args.exp_name}-test_acc')


def test_criticalfl():
    pass


def test_ifca():
    pass


def main():
    global_logger.info(f"#####{args.exp_name}#####")

    if args.curt_mode == VState.FedAvg:
        test_fedavg()
    elif args.curt_mode == VState.FedProx:
        test_fedprox()
    elif args.curt_mode == VState.FedLA:
        test_fedla()
    elif args.curt_mode == VState.SCAFFOLD:
        test_scaffold()
    elif args.curt_mode == VState.MOON:
        test_moon()
    elif args.curt_mode == VState.CriticalFL:
        test_criticalfl()
    elif args.curt_mode == VState.IFCA:
        test_ifca()
    else:
        global_logger.info(f"#####Default#####")
        simulation_federal_process()


def test_master():
    curt_dist = torch.tensor([0., 0.])
    dataset_dist = [torch.tensor([0.5, 0.5]),
                    torch.tensor([0.9, 0.1]),
                    torch.tensor([0.7, 0.3])]

    js_distance = []
    for dist in dataset_dist:
        js_distance.append(js_divergence(curt_dist, dist))

    sort_rank = np.argsort(np.array(js_distance))[::-1]
    curt_selected = sort_rank[:2]

    for ind in curt_selected:
        curt_dist += dataset_dist[ind]
