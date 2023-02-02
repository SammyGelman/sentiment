import argparse

parser = argparse.ArgumentParser(
                    prog="sentiment.py",
                    description="write string and output sentiment")

parser.add_argument('sentence', type=str,
                    help="string to test sentiment")

args = parser.parse_args()

sentence = args.sentence

print(sentence)

