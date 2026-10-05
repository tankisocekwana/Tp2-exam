"""Pure-pandas helpers for the eThekwini election explorer (no Streamlit imports)."""
import io

import numpy as np
import pandas as pd

SHORT_NAMES = {
    "AFRICAN NATIONAL CONGRESS": "ANC",
    "DEMOCRATIC ALLIANCE": "DA",
    "ECONOMIC FREEDOM FIGHTERS": "EFF",
    "INKATHA FREEDOM PARTY": "IFP",
    "AFRICAN CHRISTIAN DEMOCRATIC PARTY": "ACDP",
    "AFRICAN INDEPENDENT CONGRESS": "AIC",
    "ACTIONSA": "ActionSA",
    "INDEPENDENT": "Independent",
}

PARTY_COLORS = {
    "ANC": "#1b7f3b",
    "DA": "#2563eb",
    "EFF": "#dc2626",
    "IFP": "#f59e0b",
    "ACDP": "#7c3aed",
    "ActionSA": "#0891b2",
    "AIC": "#db2777",
    "Independent": "#6b7280",
    "Other": "#9ca3af",
}

STATION_KEYS = ["Year", "BallotType", "Ward", "VotingDistrict"]


def load_files(files):
    """files: iterable of (name, raw_bytes). Returns one tidy DataFrame."""
    frames = []
    for name, raw in files:
        d = pd.read_csv(io.BytesIO(raw), encoding="utf-8-sig")
        d.columns = d.columns.str.strip()
        d["Year"] = pd.to_datetime(d["DateGenerated"].iloc[0]).year
        d["SourceFile"] = name
        frames.append(d)
    df = pd.concat(frames, ignore_index=True)

    for c in ["Province", "Municipality", "Ward", "VotingStationName", "PartyName"]:
        df[c] = df[c].astype(str).str.strip()
    df["Party"] = df["PartyName"].map(SHORT_NAMES).fillna(df["PartyName"].str.title())
    df["WardNo"] = pd.to_numeric(df["Ward"].str[-3:], errors="coerce").astype("Int64")
    df["WardLabel"] = "Ward " + df["WardNo"].astype(str)
    df["Station"] = df["VotingStationName"].str.title() + " (" + df["VotingDistrict"].astype(str) + ")"
    return df.drop(columns=["DateGenerated"])


def station_table(d):
    """One row per voting station / ballot with registered, valid, spoilt, turnout."""
    base = d.drop_duplicates(STATION_KEYS)[
        STATION_KEYS + ["WardLabel", "WardNo", "Station", "VotingStationName", "RegisteredVoters", "SpoiltVotes"]
    ]
    valid = d.groupby(STATION_KEYS)["TotalValidVotes"].sum().rename("ValidVotes").reset_index()
    out = base.merge(valid, on=STATION_KEYS)
    out["VotesCast"] = out["ValidVotes"] + out["SpoiltVotes"]
    out["Turnout"] = np.where(out["RegisteredVoters"] > 0, out["VotesCast"] / out["RegisteredVoters"] * 100, np.nan)
    return out


def summary(d):
    """Headline numbers for a (single year, single ballot) slice."""
    st_ = station_table(d)
    reg, valid, spoilt = st_["RegisteredVoters"].sum(), st_["ValidVotes"].sum(), st_["SpoiltVotes"].sum()
    return {
        "stations": len(st_),
        "wards": d["Ward"].nunique(),
        "registered": int(reg),
        "valid": int(valid),
        "spoilt": int(spoilt),
        "turnout": (valid + spoilt) / reg * 100 if reg else np.nan,
        "spoilt_pct": spoilt / (valid + spoilt) * 100 if (valid + spoilt) else np.nan,
    }


def party_totals(d, by=None):
    """Votes and % share per party (optionally within groups `by`)."""
    by = by or []
    v = d.groupby(by + ["Party"], as_index=False)["TotalValidVotes"].sum()
    tot = v.groupby(by)["TotalValidVotes"].transform("sum") if by else v["TotalValidVotes"].sum()
    v["Share"] = np.where(tot > 0, v["TotalValidVotes"] / tot * 100, np.nan)
    return v.sort_values(by + ["TotalValidVotes"], ascending=[True] * len(by) + [False]).reset_index(drop=True)


def winners(d, by):
    """Winner, runner-up and margin for each group in `by`."""
    v = party_totals(d, by)
    v = v[v.groupby(by)["TotalValidVotes"].transform("sum") > 0].copy()
    v["Rank"] = v.groupby(by)["TotalValidVotes"].rank(method="first", ascending=False)
    top = v[v["Rank"] == 1].rename(columns={"Party": "Winner", "Share": "WinnerShare", "TotalValidVotes": "WinnerVotes"})
    sec = v[v["Rank"] == 2].rename(columns={"Party": "RunnerUp", "Share": "RunnerUpShare"})
    out = top[by + ["Winner", "WinnerVotes", "WinnerShare"]].merge(sec[by + ["RunnerUp", "RunnerUpShare"]], on=by, how="left")
    out["Margin"] = out["WinnerShare"] - out["RunnerUpShare"].fillna(0)
    return out


def group_small(v, min_share, share_col="Share", votes_col="TotalValidVotes", name_col="Party"):
    """Collapse parties below `min_share` % into 'Other'."""
    big = v[v[share_col] >= min_share]
    small = v[v[share_col] < min_share]
    if len(small):
        other = pd.DataFrame({name_col: ["Other"], votes_col: [small[votes_col].sum()], share_col: [small[share_col].sum()]})
        big = pd.concat([big, other], ignore_index=True)
    return big


def compare_parties(d, year_a, year_b):
    a = party_totals(d[d["Year"] == year_a]).set_index("Party")
    b = party_totals(d[d["Year"] == year_b]).set_index("Party")
    out = pd.DataFrame(
        {
            f"Votes {year_a}": a["TotalValidVotes"],
            f"Votes {year_b}": b["TotalValidVotes"],
            f"Share {year_a}": a["Share"],
            f"Share {year_b}": b["Share"],
        }
    ).fillna(0)
    out["Change (pp)"] = out[f"Share {year_b}"] - out[f"Share {year_a}"]
    out["Votes change"] = out[f"Votes {year_b}"] - out[f"Votes {year_a}"]
    return out.sort_values(f"Share {year_b}", ascending=False).reset_index()


def station_swing(d, party, year_a, year_b):
    """Match voting districts present in both years and compare a party's share."""
    def one(y):
        s = d[d["Year"] == y]
        tot = s.groupby("VotingDistrict")["TotalValidVotes"].sum().rename("Valid")
        p = s[s["Party"] == party].groupby("VotingDistrict")["TotalValidVotes"].sum().rename("PartyVotes")
        names = s.drop_duplicates("VotingDistrict").set_index("VotingDistrict")[["Station", "WardLabel"]]
        x = pd.concat([names, tot, p], axis=1).fillna({"PartyVotes": 0})
        x = x[x["Valid"] > 0]
        x["Share"] = x["PartyVotes"] / x["Valid"] * 100
        return x

    a, b = one(year_a), one(year_b)
    m = a[["Station", "Share", "Valid"]].join(b[["WardLabel", "Share", "Valid"]], how="inner", lsuffix=f"_{year_a}", rsuffix=f"_{year_b}")
    m["Swing (pp)"] = m[f"Share_{year_b}"] - m[f"Share_{year_a}"]
    return m.reset_index()