terraform {
  backend "s3" {
    bucket = "infinite-hashes-noqfmo"
    key    = "prod/main.tfstate"
    region = "us-east-1"
  }
}
