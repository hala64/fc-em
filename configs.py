from yacs.config import CfgNode as CN

_C = CN()
# -----------------------------------------------------------------------------
# EXPERIMENT
# -----------------------------------------------------------------------------
_C.EXP = CN()
_C.EXP.EXP_ID = ""
_C.EXP.SEED = 0
_C.EXP.TASK = 'cls'
_C.EXP.DATASET = 'modelnet40_dgcnn'
_C.EXP.MODEL_NAME = 'pointnet'
_C.EXP.LOSS_NAME = 'cross_entropy'
_C.EXP.OPTIMIZER = 'vanilla'
_C.EXP.METRIC = 'acc'
_C.EXP.class_wise = False
_C.EXP.poison_type = None
#------------------------------------------------------------------------------
# Extra Experiment Parameters
#------------------------------------------------------------------------------
_C.EXP_EXTRA = CN()
_C.EXP_EXTRA.no_val = False
_C.EXP_EXTRA.no_test = False
_C.EXP_EXTRA.robust_test = False
_C.EXP_EXTRA.val_eval_freq = 1
_C.EXP_EXTRA.test_eval_freq = 1
_C.EXP_EXTRA.save_ckp = 25
_C.EXP_EXTRA.get_feature = 'none'
_C.EXP_EXTRA.no_dropout = False
_C.EXP_EXTRA.get_fc_loss = False
_C.EXP_EXTRA.get_ce_loss = False
_C.EXP_EXTRA.get_distance = False
_C.ADVT = CN()
_C.ADVT.NAME = 'no'
_C.EXP_EXTRA.temperature = 0.1
_C.EXP_EXTRA.eps = 0.1
_C.EXP_EXTRA.chamf_coeff = 3.0
_C.EXP_EXTRA.get_model_similarity = False

# -----------------------------------------------------------------------------
# DATALOADER (contains things common across the datasets)
# -----------------------------------------------------------------------------
_C.DATALOADER = CN()
_C.DATALOADER.batch_size = 60
_C.DATALOADER.num_workers = 0
_C.DATALOADER.poison_path = None
_C.DATALOADER.poison_gen = None
# -----------------------------------------------------------------------------
# TRAINING DETAILS (contains things common across the training)
# -----------------------------------------------------------------------------
_C.TRAIN = CN()
_C.TRAIN.num_epochs = 300
_C.TRAIN.learning_rate = 1e-3
_C.TRAIN.lr_decay_factor = 0.5
_C.TRAIN.lr_reduce_patience = 10
_C.TRAIN.l2 = 0.0
_C.TRAIN.early_stop = 300
_C.TRAIN.lr_clip = 0.00001


# SCANOBJECTNN
#-----------------------------------------------------------------------------
_C.DATALOADER.SCANOBJECTNN = CN()

#-----------------------------------------------------------------------------
# MODELNET40
#-----------------------------------------------------------------------------
_C.DATALOADER.MODELNET40_DGCNN = CN()
_C.DATALOADER.MODELNET40_DGCNN.train_data_path = './data/modelnet40_ply_hdf5_2048/train_files.txt'
_C.DATALOADER.MODELNET40_DGCNN.valid_data_path = './data/modelnet40_ply_hdf5_2048/train_files.txt'
_C.DATALOADER.MODELNET40_DGCNN.test_data_path  = './data/modelnet40_ply_hdf5_2048/test_files.txt'
_C.DATALOADER.MODELNET40_DGCNN.num_points      = 1024

# ----------------------------------------------------------------------------
# MODEL
# -----------------------------------------------------------------------------
_C.MODEL = CN()

# PN2 MODEL
# -----------------------------------------------------------------------------
_C.MODEL.PN2 = CN()
_C.MODEL.PN2.version_cls = 1.0

_C.AUG = CN()
_C.AUG.NAME = 'none'
_C.AUG.BETA = 1.
_C.AUG.PROB = 0.5
_C.AUG.MIXUPRATE = 0.4


_C.POISON = None
_C.POISON_EVAL = False

def get_cfg_defaults():
  return _C.clone()