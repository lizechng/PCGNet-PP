from fire import Fire
import src

if __name__ == '__main__':
    Fire({
        'train': src.train.train,
        'train_det': src.train_det.train,
        'train_occ': src.train_occ.train,
        'infer': src.infer.infer,
        'infer_det': src.infer_det.infer,
        'infer_occ': src.infer_occ.infer,
    })
