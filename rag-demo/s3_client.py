import os
import uuid
import boto3

BUCKET_NAME = os.getenv("AWS_S3_BUCKET_NAME")
AWS_REGION = os.getenv("AWS_REGION")

s3_client = boto3.client(
    "s3",
    region_name=AWS_REGION,
    aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
    aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
)


def upload_image_to_s3(file_bytes: bytes, filename: str, content_type: str) -> str:
    """
    Uploads image bytes to S3 and returns the S3 object key.
    """
    ext = filename.split(".")[-1]
    key = f"images/{uuid.uuid4()}.{ext}"

    s3_client.put_object(
        Bucket=BUCKET_NAME,
        Key=key,
        Body=file_bytes,
        ContentType=content_type,
    )

    return key


def get_image_url(key: str, expires_in: int = 3600) -> str:
    """
    Generates a temporary signed URL to access the image.
    """
    return s3_client.generate_presigned_url(
        "get_object",
        Params={"Bucket": BUCKET_NAME, "Key": key},
        ExpiresIn=expires_in,
    )
