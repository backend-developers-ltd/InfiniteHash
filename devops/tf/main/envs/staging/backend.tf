terraform {
  backend "s3" {
    bucket = "infinite-hashes-noqfmo"
    key    = "staging/main.tfstate"
    region = "us-east-1"
  }
}
