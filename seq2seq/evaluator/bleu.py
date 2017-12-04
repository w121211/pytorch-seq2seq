# code borrowed from https://github.com/google/seq2seq/blob/master/seq2seq/metrics/bleu.py

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
import re
import subprocess
import tempfile
import logging
import numpy as np

from six.moves import urllib


def moses_multi_bleu(hypotheses, references, lowercase=False):
    """Calculate the bleu score for hypotheses and references
    using the MOSES ulti-bleu.perl script.
    Args:
      hypotheses: A numpy array of strings where each string is a single example.
      references: A numpy array of strings where each string is a single example.
      lowercase: If true, pass the "-lc" flag to the multi-bleu script
    Returns:
      The BLEU score as a float32 value.
    """

    if np.size(hypotheses) == 0:
        return np.float32(0.0)

    # Get MOSES multi-bleu script
    try:
        multi_bleu_path, _ = urllib.request.urlretrieve(
            "https://raw.githubusercontent.com/moses-smt/mosesdecoder/"
            "master/scripts/generic/multi-bleu.perl")
        os.chmod(multi_bleu_path, 0o755)
    except:  # pylint: disable=W0702
        logging.error("Unable to fetch multi-bleu.perl script, using local.")
        metrics_dir = os.path.dirname(os.path.realpath(__file__))
        bin_dir = os.path.abspath(os.path.join(metrics_dir, "..", "..", "bin"))
        multi_bleu_path = os.path.join(bin_dir, "tools/multi-bleu.perl")

    # Dump hypotheses and references to tempfiles
    hypothesis_file = tempfile.NamedTemporaryFile()
    hypothesis_file.write("\n".join(hypotheses).encode("utf-8"))
    hypothesis_file.write(b"\n")
    hypothesis_file.flush()
    reference_file = tempfile.NamedTemporaryFile()
    reference_file.write("\n".join(references).encode("utf-8"))
    reference_file.write(b"\n")
    reference_file.flush()

    # Calculate BLEU using multi-bleu script
    with open(hypothesis_file.name, "r") as read_pred:
        bleu_cmd = [multi_bleu_path]
        if lowercase:
            bleu_cmd += ["-lc"]
        bleu_cmd += [reference_file.name]
        try:
            bleu_out = subprocess.check_output(
                bleu_cmd, stdin=read_pred, stderr=subprocess.STDOUT)
            bleu_out = bleu_out.decode("utf-8")
            bleu_score = re.search(r"BLEU = (.+?),", bleu_out).group(1)
            bleu_score = float(bleu_score)
        except subprocess.CalledProcessError as error:
            if error.output is not None:
                logging.warning("multi-bleu.perl script returned non-zero exit code")
                logging.warning(error.output)
            bleu_score = np.float32(0.0)

    # Close temp files
    hypothesis_file.close()
    reference_file.close()

    return np.float32(bleu_score)
