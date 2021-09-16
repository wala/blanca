
import sys
from utils.util import evaluate_classification
import argparse
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Hierarchy prediction based on embeddings')
    parser.add_argument('--eval_file', type=str,
                        help='train/test file')
    parser.add_argument('--embed_type', type=str,
                        help='USE or bert or roberta or finetuned or bertoverflow')
    parser.add_argument('--model_dir', type=str,
                        help='dir for finetuned or bertoverflow models', required=False)
    args = parser.parse_args()

    # dataSetPath = sys.argv[1]
    # embed_type = sys.argv[2]
    # model_path = sys.argv[3]
    evaluate_classification(args.embed_type, args.model_dir, args.eval_file, 'docstring', 'text', 'label', 1)
