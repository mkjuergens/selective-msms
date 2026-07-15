from massspecgym.data.datasets import RetrievalDataset
import numpy as np
import torch

def candidate_fps_to_dense(candidate_fps, n_candidates: int, fp_size: int = 4096):
    """Return dense candidate fingerprints from dense or packed helper arrays."""
    arr = np.asarray(candidate_fps)
    if arr.ndim == 2:
        if arr.shape[0] != n_candidates:
            raise ValueError(
                f"Candidate fingerprint rows ({arr.shape[0]}) do not match "
                f"candidate labels ({n_candidates})."
            )
        if arr.shape[1] != fp_size:
            raise ValueError(f"Expected candidate fingerprints with {fp_size} bits, got {arr.shape[1]}.")
        return arr.astype(bool, copy=False)

    if arr.ndim == 1 and arr.dtype == np.uint8:
        n_bits = n_candidates * fp_size
        expected_bytes = (n_bits + 7) // 8
        if arr.size < expected_bytes:
            raise ValueError(
                f"Packed candidate fingerprints are too short: got {arr.size} bytes, "
                f"need at least {expected_bytes} for {n_candidates} candidates."
            )
        bits = np.unpackbits(arr[:expected_bytes], bitorder="big")[:n_bits]
        return bits.reshape(n_candidates, fp_size).astype(bool, copy=False)

    raise ValueError(
        "Unsupported candidate fingerprint helper shape/dtype: "
        f"shape={arr.shape}, dtype={arr.dtype}."
    )


def _normalize_inchikey(value):
    if isinstance(value, bytes):
        value = value.decode()
    return str(value).split("-")[0]


class RetrievalDataset_PrecompFPandInchi(RetrievalDataset):
    def __init__(
        self,
        fp_pth = None,
        inchi_pth = None,
        candidates_fp_pth = None,
        candidates_inchi_pth = None,
        label_mode: str = "fingerprint",
        query_identity_source: str = "precomputed",
        missing_target_policy: str = "error",
        lazy_candidate_helpers: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if label_mode not in {"fingerprint", "inchikey", "inchikey_fallback"}:
            raise ValueError("label_mode must be one of: fingerprint, inchikey, inchikey_fallback")
        if query_identity_source not in {"precomputed", "tsv"}:
            raise ValueError("query_identity_source must be one of: precomputed, tsv")
        if missing_target_policy not in {"error", "allow"}:
            raise ValueError("missing_target_policy must be one of: error, allow")
        self.label_mode = label_mode
        self.query_identity_source = query_identity_source
        self.missing_target_policy = missing_target_policy
        fp_values = np.load(fp_pth)
        self.fp_size = int(fp_values.shape[1])
        self.metadata["fp_4096"] = list(fp_values)
        precomputed_inchikeys = list(np.load(inchi_pth))
        self.metadata["inchikey_precomputed"] = precomputed_inchikeys
        if query_identity_source == "precomputed":
            self.metadata["inchikey"] = precomputed_inchikeys
        elif "inchikey" not in self.metadata.columns:
            raise ValueError("query_identity_source='tsv' requires an inchikey column in the dataset TSV")
        
        self.lazy_candidate_helpers = bool(lazy_candidate_helpers)
        self.candidates_fp_pth = str(candidates_fp_pth)
        self.candidates_inchi_pth = str(candidates_inchi_pth)
        self.candidate_fps = None if self.lazy_candidate_helpers else dict(np.load(candidates_fp_pth))
        self.candidate_inchi = None if self.lazy_candidate_helpers else dict(np.load(candidates_inchi_pth))

    def _ensure_candidate_helpers(self):
        if self.candidate_fps is None:
            self.candidate_fps = np.load(self.candidates_fp_pth)
        if self.candidate_inchi is None:
            self.candidate_inchi = np.load(self.candidates_inchi_pth)

    def __getitem__(self, i):
        self._ensure_candidate_helpers()
        item = super(RetrievalDataset, self).__getitem__(i, transform_mol=False)

        # Save the original SMILES representation of the query molecule (for evaluation)
        item["smiles"] = item["mol"]

        # Get candidates
        if item["mol"] not in self.candidates:
            raise ValueError(f'No candidates for the query molecule {item["mol"]}.')
        item["candidates"] = self.candidates[item["mol"]]

        # Save the original SMILES representations of the canidates (for evaluation)
        item["candidates_smiles"] = item["candidates"]
        item["query_inchikey"] = _normalize_inchikey(self.metadata["inchikey"].iloc[i])
        candidate_inchi = self.candidate_inchi[item["smiles"]]
        item["candidates_inchi"] = [
            _normalize_inchikey(candidate)
            for candidate in candidate_inchi.tolist()
        ]



        # Transform the query and candidate molecules
        item["mol"] = self.metadata["fp_4096"].iloc[i].astype(np.int32)
        item["candidates"] = candidate_fps_to_dense(
            self.candidate_fps[item["smiles"]],
            n_candidates=len(item["candidates_inchi"]),
            fp_size=self.fp_size,
        )
        if isinstance(item["mol"], np.ndarray):
            item["mol"] = torch.as_tensor(item["mol"], dtype=self.dtype)
        if isinstance(item["candidates"], np.ndarray):
            item["candidates"] = torch.as_tensor(item["candidates"], dtype=self.dtype)

        if self.label_mode in {"inchikey", "inchikey_fallback"}:
            item["labels"] = [
                c == item["query_inchikey"]
                for c in item["candidates_inchi"]
            ]
            if self.label_mode == "inchikey_fallback" and not any(item["labels"]):
                item["labels"] = [
                    (c == item["mol"]).all().item() for c in item["candidates"]
                ]
        else:
            item["labels"] = [
                (c == item["mol"]).all().item() for c in item["candidates"]
            ]

        item["target_present"] = bool(any(item["labels"]))
        if not item["target_present"] and self.missing_target_policy == "error":
            raise ValueError(
                f'Query molecule {item["mol"]} not found in the candidates list.'
            )

        return item
