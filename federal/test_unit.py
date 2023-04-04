import torch.cuda

from env.running_env import args, global_logger
from env.support_config import VState
from federal.federal_util import simulation_federal_process, get_data_ratio, simulation_federal_process_gan
from federal.simulation.Master import FedAvgMaster, FedProxMaster, FedIRMaster, CALIMFLMaster, HRankFLMaster


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


def test_fedir():
    loader, loaders, user_dict = simulation_federal_process()
    global_dist, device_ratios = get_data_ratio(user_dict)
    loss_weights = []

    for ratio in device_ratios:
        loss_weight = ratio / global_dist
        loss_weight = torch.where(torch.isinf(loss_weight), torch.full_like(loss_weight, 10.0), loss_weight)
        loss_weights.append(loss_weight)

    master_node = FedIRMaster(workers=args.workers, activists=args.active_workers, local_epoch=args.local_epoch,
                              loader=loader, workers_loaders=loaders, loss_weights=loss_weights)
    master_node.union_run(args.federal_round)
    master_node.cell.exit_proc(one_key=f'{args.exp_name}-test_acc')


def test_hrankfl():
    loader, loaders, user_dict = simulation_federal_process()
    master_node = HRankFLMaster(workers=args.workers, activists=args.active_workers,
                                local_epoch=args.local_epoch, loader=loader,
                                workers_loaders=loaders)
    master_node.union_run(args.federal_round)
    master_node.cell.exit_proc(one_key=f'{args.exp_name}-test_acc')


def test_calimfl(gan: bool = False):
    loader, loaders, user_dict = simulation_federal_process()
    if gan:
        aug_loaders = simulation_federal_process_gan(user_dict)
    else:
        aug_loaders = dict()
        for i in range(args.workers):
            aug_loaders[i] = None
    master_node = CALIMFLMaster(workers=args.workers, activists=args.active_workers,
                                local_epoch=args.local_epoch, loader=loader,
                                workers_loaders=loaders, aug_loaders=aug_loaders)
    master_node.prune_init(args.prune_rate, args.check_inter, args.random_data)
    master_node.union_run(args.federal_round)
    master_node.cell.exit_proc(one_key=f'{args.exp_name}-test_acc')


def main():
    global_logger.info(f"#####{args.exp_name}#####")

    if args.curt_mode == VState.FedAvg:
        test_fedavg()
    elif args.curt_mode == VState.FedProx:
        test_fedprox()
    elif args.curt_mode == VState.FedIR:
        test_fedir()
    elif args.curt_mode == VState.HRankFL:
        test_hrankfl()
    elif args.curt_mode == VState.CALIMFL:
        test_calimfl(gan=False)
    else:
        global_logger.info(f"#####Default#####")
        simulation_federal_process()
