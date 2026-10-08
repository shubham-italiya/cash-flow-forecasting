# Lambda container image (TensorFlow is too large for a zip deployment). Build from the repository root:
#   docker build -f deploy/Dockerfile -t cashflow-forecast .
FROM public.ecr.aws/lambda/python:3.12

COPY deploy/requirements.txt ${LAMBDA_TASK_ROOT}/
RUN pip install --no-cache-dir -r ${LAMBDA_TASK_ROOT}/requirements.txt

COPY src/cashflow ${LAMBDA_TASK_ROOT}/cashflow
COPY deploy/lambda_function.py ${LAMBDA_TASK_ROOT}/

CMD ["lambda_function.handler"]
