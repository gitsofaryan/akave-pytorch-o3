from pytorch_o3.client import O3Client

BUCKET_NAME = "test bucket"

def main():
    client = O3Client()

    print("Creating bucket")
    try:
        client.create_bucket(BUCKET_NAME)
        print("Bucket created")
    except Exception as e:
        print("Bucket may already exist", e)

    print("Listing buckets")
    buckets = client.list_buckets()
    print("Buckets:", buckets)

    if BUCKET_NAME in str(buckets):
        print("Verification successful")
    else:
        print("Verification failed")

    client.close()


if __name__ == "__main__":
    main()