from maformer import MAFormer

def build_models(num_classes):
    Net_S = MAFormer(num_classes=num_classes)
    Net_T = MAFormer(num_classes=num_classes)
    Net_P = MAFormer(num_classes=num_classes)
    return Net_S, Net_T, Net_P