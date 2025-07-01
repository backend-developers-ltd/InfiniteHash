terraform {
  backend "s3" {
    bucket = "luxor-subnet-noqfmo"
    key    = "prod/main.tfstate"
    region = "us-east-1"
  }
}
