CLASSES = [
    "Abi",
    "Ast",
    "Baba",
    "Bakht",
    "Diruz",
    "Dislike",
    "Faramush",
    "Kheili",
    "Khosh",
    "Like",
    "Maman",
    "Omidvar",
    "Ruz",
    "Saal",
    "Sabz",
    "Tabestun"
]
CLASSES2INDEX = {k:v for v,k in enumerate(CLASSES)}
INDEX2CLASSES = {v:k for v,k in enumerate(CLASSES)}
BLANK = len(CLASSES)

FEATURES = [
  "acc_x",
  "acc_y",
  "acc_z",
  "line_acc_x",
  "line_acc_y",
  "line_acc_z",
  "gyro_x",
  "gyro_y",
  "gyro_z",
  "gravity_x",
  "gravity_y",
  "gravity_z",
  "flex_0",
  "flex_1",
  "flex_2",
  "flex_3",
  "flex_4"
]
FEATURES2INDEX = {k:v for v,k in enumerate(FEATURES)}
INDEX2FEATURES = {v:k for v,k in enumerate(FEATURES)}

OUTLIER_BT = {'acc_x': 100, 'acc_z': 100, 'gyro_z': 20, 'gravity_y': 11}
OUTLIER_LT = {'acc_x': -100, 'acc_z': -100}