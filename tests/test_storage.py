from hidream_o1.storage import StorageConfig, build_object_key


def test_storage_config_reads_s3_compatible_environment():
    env = {
        "S3_ENDPOINT_URL": "https://storage.example.com",
        "S3_REGION": "auto",
        "S3_BUCKET": "images",
        "S3_ACCESS_KEY_ID": "key",
        "S3_SECRET_ACCESS_KEY": "secret",
        "S3_PUBLIC_BASE_URL": "https://cdn.example.com/generated",
    }

    config = StorageConfig.from_env(env)

    assert config is not None
    assert config.bucket == "images"
    assert config.public_base_url == "https://cdn.example.com/generated"
    assert config.public_url("hidream-o1/job-1.png") == "https://cdn.example.com/generated/hidream-o1/job-1.png"


def test_storage_config_returns_none_when_required_values_missing():
    assert StorageConfig.from_env({}) is None


def test_build_object_key_includes_prefix_job_and_extension():
    key = build_object_key("job-123", "webp", "hidream-o1")

    assert key.startswith("hidream-o1/job-123-")
    assert key.endswith(".webp")
