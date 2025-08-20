"""
recommender.py
Runtime scorer for Nutri Recommender (Foods → Labs deltas → Reward).

Usage:
    rec = NutriRecommender(model_dir="models/FinalModel")
    foods = pd.read_parquet("data/processed/foods_dictionary_plus_enriched.parquet")
    labs  = {"ldl": 130, "hdl": 35, "triglycerides": 180, "fasting_glucose": 105}
    top   = rec.recommend(foods, labs, top_k=50, strict_mode=True)
"""

from __future__ import annotations
import json, os, re, unicodedata
from pathlib import Path
from typing import Dict, List, Optional, Union

import joblib
import numpy as np
import pandas as pd


# =============================== Paths / Loading ===============================

def _default_model_dir() -> Path:
    # Environment override or repo default
    return Path(os.getenv("MODEL_DIR", "models/FinalModel"))

# Vitamin D & Ferritin aliases you may have used in training
VITD_CANDIDATES     = ["LBXVIDMS","LBXVID","LBXVIDLC","VIDMS","VID","LBDVIDMS"]
FERRITIN_CANDIDATES = ["LBXFER","LBDFERSI","LBDFERS","FER","FERN"]

# Human → NHANES mapping (stable, used by API callers)
HUMAN_TO_NHANES = {
    # Metabolic
    "fasting_glucose": "LBXSGL",
    "cholesterol_total": "LBXSCH",
    "hdl": "LBDHDL",
    "ldl": "LBDLDL",
    "triglycerides": "LBXSTR",
    "uric_acid": "LBXSUA",
    # Renal / Electrolytes
    "creatinine": "LBXSCR",
    "bun": "LBXSBU",
    "calcium": "LBXSCA",
    "sodium": "LBXSNA",
    "potassium": "LBXSKSI",
    "chloride": "LBXCLSI",
    # Note: some datasets use LBXSAPSI for CO2/ALP; keep as-is to match your artifacts
    "alk_phos": "LBXSAPSI",
    # Liver
    "ast": "LBXAST",
    "alt": "LBXALT",
    "albumin": "LBXSAL",
    "total_protein": "LBXTP",
    "total_bilirubin": "LBXTB",
    # Hematology
    "hemoglobin": "LBXHGB",
    "hematocrit": "LBXHCT",
    "rbc": "LBXRBCSI",
    "mcv": "LBXMCVSI",
    "mch": "LBXMCHSI",
    "mchc": "LBXMCHCSI",
    "rdw": "LBXRDW",
    # Optional
    "vitamin_d_25oh": "LBXVIDMS",
    "ferritin": "LBXFER",
}
NHANES_KEYS = set(HUMAN_TO_NHANES.values())
HUMAN_KEYS  = set(HUMAN_TO_NHANES.keys())


# ----------- Clinical ranges (units matching your training data) --------------
RANGES: Dict[str, tuple] = {
    "LBXSGL": (70, 99),
    "LBXSCH": (100, 199),
    "LBDHDL": (40, 90),
    "LBDLDL": (0, 100),
    "LBXSTR": (0, 150),
    "LBXSUA": (3.5, 7.2),
    "LBXSCR": (0.6, 1.3),
    "LBXSBU": (7, 20),
    "LBXSCA": (8.6, 10.2),
    "LBXSNA": (135, 145),
    "LBXSKSI": (3.5, 5.1),
    "LBXCLSI": (98, 107),
    "LBXSAPSI": (22, 29),   # NOTE: kept for compatibility with your artifacts
    "LBXAST": (0, 40),
    "LBXALT": (0, 44),
    "LBXSAL": (3.4, 5.4),
    "LBXTP": (6.0, 8.3),
    "LBXTB": (0.1, 1.2),
    "LBXHGB": (13.5, 17.5),
    "LBXHCT": (38.8, 50.0),
    "LBXRBCSI": (4.5, 5.9),
    "LBXMCVSI": (80, 100),
    "LBXMCHSI": (27, 33),
    "LBXMCHCSI": (32, 36),
    "LBXRDW": (11.5, 14.5),
    "LBXVIDMS": (30, 60),
    "LBXFER": (30, 300),
}


# =============================== Junk Filters (v4) =============================

def _normalize_text(x: str) -> str:
    if x is None:
        return ""
    x = unicodedata.normalize("NFKC", str(x))
    x = x.replace("\u00A0", " ").replace("\u00AD", "")
    x = re.sub(r"\s+", " ", x).strip().lower()
    return x

def _normalize_series(s: pd.Series) -> pd.Series:
    s = s.astype(str).fillna("")
    return s.map(_normalize_text)

ALCOHOL_PAT = re.compile(
    r"\b("
    r"beer|ale|lager|stout|ipa|cider|mead|malt\s+beverage|alcoholic\s+malt\s+beverage|"
    r"wine|red\s+wine|white\s+wine|ros[eé]|champagne|prosecco|port|sherry|vermouth|sangria|"
    r"vodka|whiskey|whisky|bourbon|rye|rum|tequila|mezcal|gin|brandy|cognac|liqueur|schnapps|"
    r"jaeger|jäger|jager|jagerbomb|jägerbomb|alexander|"
    r"cocktail|mixed\s*drink|martini|manhattan|negroni|old\s*fashioned|margarita|paloma|mojito|"
    r"daiquiri|gimlet|cosmopolitan|cosmo|mule|moscow\s*mule|spritz|aperol\s*spritz|"
    r"long\s*island\s*iced\s*tea|white\s*russian|black\s*russian|kamikaze|orange\s*blossom|"
    r"tom\s*collins|whiskey\s*sour|amaretto\s*sour|pi[nñ]a\s*colada|mai\s*tai|mint\s*julep"
    r")\b", re.I,
)
BRAND_JUNK = re.compile(
    r"\b("
    r"cheerios|honey\s+nut\s+cheerios|multigrain\s+cheerios|frosted\s+cheerios|berry\s+burst\s+cheerios|"
    r"kix|honey\s+kix|berry\s+berry\s+kix|lucky\s+charms|wheaties|life(\b|[^a-z])|alpha[-\s]*bits|quisp|"
    r"crispix|king\s+vitaman|frosty\s*o'?s|oh'?s(\b|[^a-z])|cracklin'?[-\s]*oat[-\s]*bran|basic\s*4|"
    r"chex(\b|[^a-z])|rice\s+chex|corn\s+chex|wheat\s+chex|chex\s+cinnamon|corn\s+pops|golden\s+crisp|"
    r"fruity\s+pebbles|cocoa\s+pebbles|cocoa\s+krispies|fiber\s*one(\s+caramel\s+delight|\s+honey\s+clusters)?|"
    r"fruit\s*&\s*fibre|fruit\s+and\s+fibre|product\s*19|nature'?s\s+path|optimum|post\b|kellogg'?s|general\s+mills|quaker\b|malt[-\s]*o[-\s]*meal|"
    r"skittles|butterfinger|baby\s+ruth|reese'?s\s+pieces"
    r")\b", re.I,
)
INFANT_PAT = re.compile(
    r"\b("
    r"infant\s*formula|formula\b|enfamil|pediatric|toddler|baby\s*food|"
    r"baby\b.*(puffs|snack|finger\s*food)|puff[s]?\b.*baby"
    r")\b", re.I,
)
FATS_PAT = re.compile(
    r"\b("
    r"butter|ghee|clarified\s*butter|butter[-\s]*oil\s*blend|butter[-\s]*margarine\s*blend|"
    r"shortening|lard|animal\s*fat|drippings|table\s*fat|"
    r"coconut\s*oil|coconut\s*cream|whipped\s*cream\b|cream,\s*light,\s*whipped|cream\s+substitute"
    r")\b", re.I,
)
DRY_DAIRY_PAT = re.compile(r"\b(milk|cream|yogurt|kefir|buttermilk)\b.*\b(dry|powder(?:ed)?)\b", re.I)
COATED_FRIED_PAT = re.compile(r"\b(coated|breaded|battered)\b.*\b(baked|broiled|fried)\b", re.I)
DRIED_FRUIT_SWEETS = re.compile(
    r"\b("
    r"(papaya|persimmon|pineapple|banana|apple|peach|apricot|mango|cherry|cranberr(?:y|ies)|"
    r"raisin[s]?|date[s]?|fig[s]?|pear[s]?|plum[s]?|prune[s]?|blueberr(?:y|ies)|coconut)"
    r")\b.*\b(dried|dehydrated|sweetened|candied)\b", re.I,
)
UPF_PAT = re.compile(
    r"\b("
    # beverages / energy / sodas
    r"energy\s*drink|energy\s*juice\s*drink|juice\s*drink|beverage\b|ocean\s*spray|cran[-\s]*energy|sobe|mona\s*vie|monavie|vitamin\s*water|"
    r"gatorade|powerade|monster|red\s*bull|rockstar|prime\s*hydration|5[-\s]?hour\s*energy|"
    r"soft\s*drink|soda|cola|diet\s*(soda|cola|soft\s*drink)|zero[-\s]*sugar\s*(soda|cola|soft\s*drink)|sugar[-\s]*free\s*(soda|cola|soft\s*drink)|"
    r"soft\s*drink[, ]*fruit[-\s]*flavor(?:ed)?[, ]*caffeine\s*contain(?:ing)?[, ]*sugar[-\s]*free|"

    # bars / powders / mixes
    r"energy\s*bar|protein\s*bar|meal\s*bar|nutrition(al)?\s*(drink|shake)\s*mix|powder|mix|concentrate|instant\s*(coffee|tea)?|"
    r"clif\s*bar|quest\s*bar|rxbar|kind\s*bar|powerbar|carnation\s+instant\s+breakfast|herbalife|slim\s*fast|slimfast|breakfast\s*bar|snack\s*bar|"

    # desserts/sweets/pastry
    r"beignet|cruller|baked\s*alaska|cake|cupcake|cookie|doughnut|donut|pastry|eclair|cream\s*puff|strudel|fritter|tart|pie|cobbler|"
    r"ice\s*cream|gelato|sundae|frozen\s*yogurt|pudding|custard|tembleque|haupia|harina\s+de\s+maiz\s+con\s+coco|cornmeal\s+coconut\s+dessert|"
    r"fudge|toffee|brittle|praline[s]?|candy\b|confection|"
    r"sprinkles|syrup|maple\s*syrup|molasses|agave|blue\s*agave|honey\b|"
    r"jam|jelly|preserve|marmalade|guava\s*paste|fruit\s*sauce|fruit\s*dessert|"

    # sauces/dips/condiments/gravies/toppings/dressings
    r"sauce|relish|topping|dip\b|dressing|gravy|barbecue\s*sauce|ketchup|catsup|chutney|"

    # crackers/pretzels
    r"cracker|graham|pretzel|"

    # pickles & canned fruit w/ “drained”
    r"pickle|pickles|canned.*drained|frozen.*drained|"

    # baby snacks & cough drops & gelatin
    r"baby\s*food|fruit\s*flavored\s*snack|gelatin|cough\s*drops"
    r")\b", re.I,
)
SPECIFIC_SOFT_DRINK = re.compile(
    r"\bsoft\s*drink[, ]*fruit[-\s]*flavor(?:ed)?[, ]*caffeine\s*contain(?:ing)?[, ]*sugar[-\s]*free\b",
    re.I,
)

WHOLE_FOOD_ALLOW = re.compile(
    r"\b("
    # seafood & organs
    r"salmon|sardine|herring|mackerel|trout|anchovy|cod|halibut|porgy|mullet|octopus|squid|"
    r"oyster|mussel|clam|shrimp|scallop|liver|chicken\s+liver|beef\s+liver|heart|"
    # eggs & plain dairy
    r"egg\b|eggs\b|yogurt\b(?!.*(frozen|sweet|vanilla|chocolate))|kefir\b(?!.*(sweet|vanilla|chocolate))|cottage\s+cheese\b(?!.*(sauce|spread))|"
    # leafy greens & vegetables
    r"spinach|kale|collards?|mustard\s+greens|turnip\s+greens|dandelion\s+greens|broccoli|broccoli\s+raab|chinese\s+broccoli|bok\s+choy|"
    r"chard|beet\s+greens|arugula|lettuce|brussels|cauliflower|mushroom|shiitake|oyster\s+mushroom|portobello|enoki|"
    # legumes
    r"lentil|chickpea|garbanzo|black\s+bean|kidney\s+bean|pinto|navy\s+bean|soybean|edamame|tofu|tempeh|"
    # nuts & seeds
    r"almond|walnut|pecan|pistachio|hazelnut|cashew|peanut|sunflower\s+seed|pumpkin\s+seed|sesame\s+seed|chia|flax|hemp|"
    # whole grains
    r"steel[-\s]*cut\s*oats|oat\s+groats?|oats\b|brown\s+rice|wild\s+rice|quinoa|buckwheat|millet|barley|amaranth|sorghum|teff|"
    # fresh fruit (not canned/dried)
    r"blueberr(?:y|ies)|strawberr(?:y|ies)|raspberr(?:y|ies)|blackberr(?:y|ies)|"
    r"apple\b(?!.*(dried|canned))|pear\b(?!.*(dried|canned))|"
    r"orange\b(?!.*(canned|frozen|drained))|grapefruit|clementine\b(?!.*(canned|frozen|drained))|tangerine\b(?!.*(canned|frozen|drained))|"
    r"banana\b(?!.*(chips|dried))|avocado|tomato\b(?!.*(sauce|ketchup))|pepper\b(?!.*(sauce))|cucumber|carrot|zucchini"
    r")\b", re.I,
)

def _choose_text_column(df: pd.DataFrame) -> pd.Series:
    for col in ["desc", "description", "name", "food", "canonical"]:
        if col in df.columns:
            return df[col]
    obj_cols = [c for c in df.columns if df[c].dtype == "object"]
    return df[obj_cols[0]] if obj_cols else pd.Series([""] * len(df))

def drop_blocked(df: pd.DataFrame) -> pd.DataFrame:
    """Pattern-based junk filter (no Streamlit deps)."""
    if df is None or df.empty:
        return df
    text_raw = _choose_text_column(df)
    text = _normalize_series(text_raw)
    cat  = _normalize_series(df["category"]) if "category" in df.columns else pd.Series([""] * len(df))

    bad = (
        text.str.contains(ALCOHOL_PAT, na=False)
        | text.str.contains(UPF_PAT, na=False)
        | text.str.contains(BRAND_JUNK, na=False)
        | text.str.contains(INFANT_PAT, na=False)
        | text.str.contains(FATS_PAT, na=False)
        | text.str.contains(DRIED_FRUIT_SWEETS, na=False)
        | text.str.contains(DRY_DAIRY_PAT, na=False)
        | text.str.contains(COATED_FRIED_PAT, na=False)
        | text.str.contains(SPECIFIC_SOFT_DRINK, na=False)
    )
    if len(cat) == len(text):
        bad = bad | cat.str.contains(r"alcohol", na=False)

    return df.loc[~bad].copy()

def apply_strict_whole_foods(df: pd.DataFrame) -> pd.DataFrame:
    """Allow only whitelist whole-food families."""
    if df is None or df.empty:
        return df
    text = _normalize_series(_choose_text_column(df))
    keep = text.str_contains(WHOLE_FOOD_ALLOW, na=False) if hasattr(text, "str_contains") else text.str.contains(WHOLE_FOOD_ALLOW, na=False)
    return df.loc[keep].copy()


# ================================ Utilities ==================================

def _canon_desc(s: str) -> str:
    t = re.sub(r"\s+", " ", str(s)).strip().lower()
    t = re.sub(r",?\s*ns as to.*$", "", t)
    t = re.sub(r"\s*\(.*?\)$", "", t)
    t = re.sub(r"\b(iced|hot|nonfat|flavored|decaf(?:feinated)?|regular|with.*|and.*)$", "", t).strip()
    return t

def _root_key(s: str) -> str:
    s = str(s).lower()
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"\s*\(.*?\)", "", s)
    if re.search(r"\b(coffee|espresso|latte|cappuccino|mocha|macchiato|americano|cafe|café)\b", s):
        return "coffee"
    if re.search(r"\b(tea|matcha|chai|mate)\b", s):
        return "tea"
    if re.search(
        r"\b("
        r"cereal|flakes|bran|muesli|crisp(y)?\s*rice|corn\s*bursts|corn\s*flakes|"
        r"special\s*k|product\s*19|cap'?n\s*crunch|crunch\s*berries|"
        r"total(\s|\b)|malt[-\s]*o[-\s]*meal|quaker(\s|\b)"
        r")\b", s):
        return "cereal"
    if re.search(r"\b(bar|energy\s*bar|protein\s*bar|meal\s*bar|powerbar|clif|quest|rxbar|kind)\b", s):
        return "bar"
    if re.search(r"\b(energy\s*drink|juice\s*drink|ocean\s*spray|cran[-\s]*energy|sobe|monavie|beverage)\b", s):
        return "upf_beverage"
    if re.search(r"\b(meatless|meat\s*substitute|vegetarian\b.*(fillet|patty|burger)|imitation|fish\s*stick)\b", s):
        return "imitation_meat"
    if re.search(r"\b(gravy)\b", s):
        return "gravy/sauce"
    return s.split(",")[0].strip()

def _guess_col(cols: List[str], *needles) -> Optional[str]:
    need = [n.lower() for n in needles]
    for c in cols:
        cl = c.lower()
        if all(n in cl for n in need):
            return c
    return None

def nutrient_penalty_vector(foods: pd.DataFrame) -> np.ndarray:
    """
    Per-food penalty: +added sugar, +total sugar, +sat fat, minus fiber (heavier weights).
    """
    cols = list(foods.columns)
    add_sugar = _guess_col(cols, "add", "sugar") or _guess_col(cols, "nutr", "add", "sugar")
    tot_sugar = _guess_col(cols, "total", "sugar") or _guess_col(cols, "nutr", "sugar")
    sat_fat   = _guess_col(cols, "sat", "fat") or _guess_col(cols, "fa", "sat")
    fiber     = _guess_col(cols, "fiber") or _guess_col(cols, "dfib")

    def _col(c):
        if c and c in foods.columns:
            return pd.to_numeric(foods[c], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        return np.zeros(len(foods), dtype=float)

    add_sug = _col(add_sugar)
    sug     = _col(tot_sugar)
    sat     = _col(sat_fat)
    fib     = _col(fiber)

    # Heavier penalty (per 100g):
    # 25g added sugar → 3.0; 20g sugar → 1.6; 10g sat fat → 0.8; 10g fiber → -0.3 bonus
    penalty = 0.12 * np.maximum(add_sug, 0) + 0.08 * np.maximum(sug, 0) + 0.08 * np.maximum(sat, 0) - 0.03 * np.maximum(fib, 0)
    return penalty

def _target_weights_from_user(y_user: np.ndarray, targets: List[str]) -> np.ndarray:
    """
    Weight markers by deviation from healthy band.
    Inside range -> 1.0; Outside -> 1 + beta * normalized distance.
    """
    beta = 3.0
    w = np.ones(len(targets), dtype=float)
    for k, t in enumerate(targets):
        lo, hi = RANGES.get(t, (None, None))
        if lo is None or hi is None:
            continue
        y = float(y_user[k])
        span = max(hi - lo, 1e-6)
        if y < lo:
            w[k] = 1.0 + beta * (lo - y) / span
        elif y > hi:
            w[k] = 1.0 + beta * (y - hi) / span
        else:
            w[k] = 1.0
    # normalize so total weight ≈ #targets
    w *= (len(targets) / max(w.sum(), 1e-6))
    return w

def reward_from_ranges_weighted(y_pred: np.ndarray,
                                targets: List[str],
                                weights: np.ndarray) -> np.ndarray:
    """
    +1 if inside range; otherwise linearly penalize by normalized distance.
    Weighted sum across targets.
    """
    n = y_pred.shape[0]
    total = np.zeros(n, dtype=float)
    wsum = max(weights.sum(), 1e-6)
    for k, t in enumerate(targets):
        lo, hi = RANGES.get(t, (None, None))
        if lo is None or hi is None:
            continue
        yp = y_pred[:, k]
        inside = (yp >= lo) & (yp <= hi)
        score = np.where(inside, 1.0, 0.0)
        span = max(hi - lo, 1e-6)
        dist = np.where(yp < lo, (lo - yp)/span, np.where(yp > hi, (yp - hi)/span, 0.0))
        score -= dist
        total += weights[k] * score
    return total / wsum


# =============================== Model Wrapper ===============================

class NutriRecommender:
    """
    Loads FinalModel artifacts and provides .recommend(...) for an API.
    """

    def __init__(self, model_dir: Union[str, Path, None] = None):
        self.model_dir = Path(model_dir) if model_dir else _default_model_dir()
        meta_path  = self.model_dir / "meta.json"
        scaler_path= self.model_dir / "X_scaler.joblib"
        if not meta_path.exists():
            raise FileNotFoundError(f"meta.json not found at {meta_path}")
        if not scaler_path.exists():
            raise FileNotFoundError(f"X_scaler.joblib not found at {scaler_path}")

        with open(meta_path, "r") as f:
            self.meta = json.load(f)
        self.nutrient_cols: List[str] = self.meta.get("nutrient_cols", [])
        self.present_targets: List[str] = [t for t in self.meta.get("targets_present", [])]

        # Load scaler and per-target models
        self.scaler = joblib.load(scaler_path)
        self.models = {}
        missing = []
        for t in self.present_targets:
            p = self.model_dir / f"lgbm_{t}.joblib"
            if p.exists():
                self.models[t] = joblib.load(p)
            else:
                missing.append(t)
        # Non-fatal if some targets are missing, but results may be weaker.

    # --------- Labs alignment helpers ---------
    def _labs_to_nhanes_row(self, labs: Union[Dict[str, float], pd.Series, pd.DataFrame]) -> pd.Series:
        """
        Accepts human-labeled dict (e.g., {'ldl': 120}), NHANES-labeled dict/Series,
        or a 1-row DataFrame. Returns a Series keyed by NHANES codes.
        """
        if labs is None:
            return pd.Series({k: np.nan for k in NHANES_KEYS})

        if isinstance(labs, pd.DataFrame):
            if labs.shape[0] == 0:
                return pd.Series({k: np.nan for k in NHANES_KEYS})
            s = labs.iloc[0]
        elif isinstance(labs, pd.Series):
            s = labs
        elif isinstance(labs, dict):
            s = pd.Series(labs)
        else:
            raise TypeError("labs must be dict / Series / 1-row DataFrame")

        # Normalize keys
        normalized = {str(k).strip().lower(): v for k, v in s.items()}

        # Determine if keys are human or NHANES
        if any(k in HUMAN_KEYS for k in normalized.keys()):
            nhanes = {HUMAN_TO_NHANES.get(k, None): normalized[k] for k in normalized.keys() if k in HUMAN_KEYS}
        else:
            # assume NHANES-like keys
            nhanes = {k.upper(): v for k, v in s.items() if str(k).upper().startswith("LB") or str(k).upper().startswith("LBD")}

        # Ensure all codes present
        out = {code: np.nan for code in NHANES_KEYS}
        out.update({k: float(v) if v is not None else np.nan for k, v in nhanes.items() if k in NHANES_KEYS})
        return pd.Series(out)

    def _align_to_present_targets(self, nhanes_row: pd.Series) -> np.ndarray:
        """
        Create the labs vector in the exact order of self.present_targets.
        Vitamin D and Ferritin aliases are handled by copying values if needed.
        """
        vitd_val = nhanes_row.get("LBXVIDMS", np.nan)
        ferr_val = nhanes_row.get("LBXFER",   np.nan)
        y = np.zeros(len(self.present_targets), dtype=float)
        for i, t in enumerate(self.present_targets):
            if t in nhanes_row.index:
                v = nhanes_row.get(t, np.nan)
            elif t in VITD_CANDIDATES:
                v = vitd_val
            elif t in FERRITIN_CANDIDATES:
                v = ferr_val
            else:
                v = np.nan
            y[i] = float(v) if pd.notna(v) else np.nan
        return y

    # --------- Food matrix / utilities ---------
    def _build_food_matrix(self, foods: pd.DataFrame) -> np.ndarray:
        X = np.zeros((len(foods), len(self.nutrient_cols)), dtype=float)
        for j, c in enumerate(self.nutrient_cols):
            if c in foods.columns:
                X[:, j] = pd.to_numeric(foods[c], errors="coerce").fillna(0.0).to_numpy()
        return X

    # --------- Public API ---------
    def recommend(
        self,
        foods: pd.DataFrame,
        labs: Union[Dict[str, float], pd.Series, pd.DataFrame],
        top_k: int = 50,
        batch_size: int = 512,
        strict_mode: bool = True,
        apply_filters: bool = True,
        baseline_nutrients: Optional[Union[Dict[str, float], pd.Series, pd.DataFrame]] = None,
        return_debug: bool = False,
    ) -> Union[pd.DataFrame, tuple[pd.DataFrame, Dict]]:
        """
        Score foods dynamically using user's labs; return top_k with scores.

        Params:
            foods: DataFrame with at least ['category','desc'] and NUTR_* columns used in training.
            labs:  dict/Series/DataFrame with either human names (e.g., 'ldl','hdl','triglycerides', etc.)
                   or NHANES codes (e.g., 'LBDLDL','LBDHDL','LBXSTR', etc.).
            top_k: number of items to return after dedupe & filters
            batch_size: scoring batch size (RAM/security)
            strict_mode: if True, keep only whole-food families
            apply_filters: if True, apply junk filters pre/post scoring
            baseline_nutrients: optional nutrients row (dict/Series/1-row DF) to add to each food; default zeros
            return_debug: if True, return (recs, debug_dict)

        Returns:
            DataFrame with columns ['category','desc','_score', ...]
            or (DataFrame, debug dict) if return_debug=True
        """
        if foods is None or foods.empty:
            return foods if not return_debug else (foods, {"reason": "empty_foods"})

        # Pre-filters
        F = foods.copy()
        if apply_filters:
            F = drop_blocked(F)
            if strict_mode:
                F = apply_strict_whole_foods(F)
            if F.empty:
                return F if not return_debug else (F, {"reason": "filtered_all_pre"})

        # Build nutrient matrix
        X_food = self._build_food_matrix(F)

        # Baseline nutrients vector
        if baseline_nutrients is None:
            base_vec = np.zeros((1, len(self.nutrient_cols)), dtype=float)
        else:
            if isinstance(baseline_nutrients, pd.DataFrame):
                s = baseline_nutrients.iloc[0]
            elif isinstance(baseline_nutrients, pd.Series):
                s = baseline_nutrients
            elif isinstance(baseline_nutrients, dict):
                s = pd.Series(baseline_nutrients)
            else:
                raise TypeError("baseline_nutrients must be dict/Series/1-row DataFrame")
            base_vec = np.zeros((1, len(self.nutrient_cols)), dtype=float)
            for i, c in enumerate(self.nutrient_cols):
                if c in s.index and pd.notna(s[c]):
                    base_vec[0, i] = float(s[c])

        # Base labs prediction (1 x T)
        base_scaled = self.scaler.transform(base_vec)
        base_pred = np.column_stack([self.models[t].predict(base_scaled) for t in self.present_targets])

        # Align labs to model targets; fallback to base_pred when missing
        nhanes_row = self._labs_to_nhanes_row(labs)
        y_user = self._align_to_present_targets(nhanes_row)
        y_user = np.where(np.isnan(y_user), base_pred[0], y_user)

        # Weights from user labs
        weights = _target_weights_from_user(y_user, self.present_targets)
        r_user  = reward_from_ranges_weighted(y_user.reshape(1, -1), self.present_targets, weights)[0]

        # Nutrient penalties (per food)
        penal = nutrient_penalty_vector(F)

        # Batch scoring: marginal effect (add food nutrients) → new labs → reward diff
        n = len(F)
        scores = np.zeros(n, dtype=float)
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            add_scaled = self.scaler.transform(base_vec + X_food[start:end])
            add_pred   = np.column_stack([self.models[t].predict(add_scaled) for t in self.present_targets])
            delta = add_pred - base_pred                      # (batch, T)
            y_new = y_user.reshape(1, -1) + delta             # (batch, T)
            r_new = reward_from_ranges_weighted(y_new, self.present_targets, weights)  # (batch,)
            scores[start:end] = r_new - r_user - penal[start:end]

        # Attach & dedupe
        F = F.copy()
        text_col = "desc" if "desc" in F.columns else _choose_text_column(F).name
        F["_score"] = scores
        F["_canon"] = F[text_col].astype(str).map(_canon_desc)
        F["_root"]  = F[text_col].astype(str).map(_root_key)

        # pick best per canonical description, then per root key
        F1 = (F.sort_values("_score", ascending=False)
                .groupby("_canon", as_index=False)
                .head(1))
        F2 = (F1.sort_values("_score", ascending=False)
                .groupby("_root", as_index=False)
                .head(1)
                .sort_values("_score", ascending=False)
                .reset_index(drop=True))

        # Post-filters
        if apply_filters:
            F2 = drop_blocked(F2)
            if strict_mode:
                F2 = apply_strict_whole_foods(F2)

        recs = F2.head(top_k).reset_index(drop=True)

        if return_debug:
            debug = {
                "present_targets": self.present_targets,
                "num_foods_input": len(foods),
                "num_foods_after_prefilter": len(F),
                "num_foods_after_postfilter": len(recs),
                "weights_mean": float(np.mean(weights)),
                "weights_max": float(np.max(weights)),
                "labs_used_vector": {t: float(y_user[i]) for i, t in enumerate(self.present_targets)},
            }
            return recs, debug
        return recs


# =============================== Example (CLI) ===============================
if __name__ == "__main__":
    # Minimal smoke test (expects local model dir + foods parquet)
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, default=str(_default_model_dir()))
    parser.add_argument("--foods", type=str, required=True, help="Path to foods parquet or CSV")
    parser.add_argument("--topk", type=int, default=50)
    parser.add_argument("--strict", action="store_true", help="Enable strict whole-food mode")
    args = parser.parse_args()

    # Load foods
    if args.foods.endswith(".parquet"):
        foods_df = pd.read_parquet(args.foods)
    else:
        foods_df = pd.read_csv(args.foods)

    # Example labs (human keys)
    labs_example = {"ldl": 135, "hdl": 38, "triglycerides": 180, "fasting_glucose": 105}

    rec = NutriRecommender(args.model_dir)
    top, dbg = rec.recommend(foods_df, labs_example, top_k=args.topk, strict_mode=args.strict, return_debug=True)
    cols = [c for c in ["category","desc","_score"] if c in top.columns]
    print(top[cols].head(15).to_string(index=False))
    print("\n[debug]", json.dumps(dbg, indent=2))

