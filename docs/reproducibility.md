# Provenance, limitations, and publication edits

Source: `Road_Damage_Detection_VSCode_150ep_b12 (1).ipynb` and accompanying `image_report (1)(1).pdf` supplied by the project owner. Historical metrics were transcribed from notebook cell 29's saved DataFrame, retaining three-decimal rounding. Code and saved outputs take precedence where report wording is inconsistent.

The core split, cleaning, augmentation, experiment definitions, and metric calculations are retained. Publication changes: cleared outputs/execution metadata; portable workspace path; recorded Ultralytics dependency; camera code moved to an explicit optional macOS example; private camera IP replaced with an environment variable; startup checks for platform, weights, camera, and frame dimensions; README, CSV results, and static validation CI added.

The camera prototype retains the original MPS/Thai voice/font assumptions and has not been tested here. GUI or speech errors may require platform-specific work. Its alerts are image-size heuristics, not calibrated distance estimates.

Dataset splitting uses filename prefix and blocks of trailing frame numbers (block size 30). It does not establish fully video-disjoint or location-disjoint evaluation. Real nighttime conditions, independent road collections, and repeated seeds remain future evaluations.

The exploratory BEST_EXP code ranks models using original test mAP50–95. Do not present this as validation-based model selection. Existing checkpoints are reused by path existence only; there is no dataset/configuration hash check. Use a new workspace for changed experiments and record package versions and dataset hashes.

The OBB TTA attempt explicitly fell back to single-scale inference in the saved logs. HBB and OBB scores use different task formulations; differences are descriptive and not a controlled causal comparison. E1/E2 also differ in multiple training settings.

During packaging only notebook schema/Python syntax validation was run. No training, dataset download, GPU execution, camera test, or independent metric reproduction was performed. The figure in assets is drawn only from historical numeric outputs, not fabricated detections.
