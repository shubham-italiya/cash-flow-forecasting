# Deploying the forecast API on AWS Lambda

The model is served by a container-image Lambda (`Dockerfile`), defined with AWS SAM (`template.yaml`).
Model files live in a **versioned** S3 bucket, so every trained model is kept and can be rolled back.
The handler is tested locally (`tests/test_lambda.py`) against a model folder instead of S3.

> This repository is not deployed. These steps need your own AWS account, the AWS CLI, SAM CLI and Docker.

```bash
python -m cashflow.train                                   # writes artifacts/model/
sam build -t deploy/template.yaml
sam deploy --guided                                        # creates the bucket and the function
aws s3 sync artifacts/model s3://<ModelBucketName>/models/latest/
aws lambda invoke --function-name <function> --payload '{"days": 7}' --cli-binary-format raw-in-base64-out out.json
```

Response (a real call to the handler with the model trained by `make train`, `{"days": 7}`):

```json
{"model": "lstm", "trained_until": "2011-12-09",
 "forecast": [{"date": "2011-12-10", "net_cash": 0.39},
              {"date": "2011-12-11", "net_cash": 30450.62},
              {"date": "2011-12-12", "net_cash": 44069.6}, "..."],
 "total": 262474.69}
```

10 December 2011 is a Saturday, when the shop is closed, so the forecast is about zero.
