terraform {
  backend "s3" {
    bucket = "luxor-subnet-noqfmo"
    key    = "staging/main.tfstate"
    region = "us-east-1"
  }
}
