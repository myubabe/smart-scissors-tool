import argparse
from scissors.gui import run_demo


def main(file_name):
    run_demo(file_name)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('