import argparse
import urllib.request



def compute_densities():
    # download input files from github
    url = "https://drive.google.com/uc?id=1WSjdP8HKwUMmBftOohr6NuOO3tje3Kxv"
    urllib.request.urlretrieve(url, "density_input_files.tar.gz")

    # unzip
    import tarfile
    with tarfile.open("density_input_files.tar.gz", "r:gz") as tar:
        tar.extractall()
    



def main():
    parser = argparse.ArgumentParser(description="Benchmark MACE organic")
    parser.add_argument(
        "--density",
        action="store_true",
        help="Whether to compute densities",
    )
    args = parser.parse_args()


    if args.density:
        compute_densities()



if __name__ == "__main__":
    main()


