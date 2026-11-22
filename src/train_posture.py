import argparse
from ultralytics import YOLO

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/posture_test")   # folder containing train/ and val/
    ap.add_argument("--name", default="posture_test_model")        # run name -> models/<name>/
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--imgsz", type=int, default=224)
    ap.add_argument("--batch", type=int, default=32)
    args = ap.parse_args()

    model = YOLO("yolov8n-cls.pt")           # small pretrained classifier (fine-tuning)
    model.train(
        data=args.data,#use the input of the args 
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=0,                            # GPU; use "cpu" if CUDA isn't working
        workers=2,                           # keep low on Windows
        project="models",#the parent folder to save output   
        name=args.name, #the children file under the parent folder that save output
        exist_ok=True,                       # reuse the folder if you rerun
    )#it will automatically find the train, val, test by folder name in the given path
    metrics = model.val(
    data=args.data,
    project="models",
    name=f"{args.name}_val",
    exist_ok=True,
) #model.val(split = "test") test on the test set, else default on val
    print("top1 accuracy:", metrics.top1)

if __name__ == "__main__":                   # required on Windows
    main()