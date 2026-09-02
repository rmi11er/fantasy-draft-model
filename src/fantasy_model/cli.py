from __future__ import annotations

import argparse
import json

from .league import MFLClient


def main() -> None:
    parser = argparse.ArgumentParser(description="Read the live MFL draft state")
    parser.add_argument("--year", type=int, default=2026)
    parser.add_argument("--league", default="50473")
    parser.add_argument("--franchise", default="Levered Leopolds")
    args = parser.parse_args()

    snapshot = MFLClient(args.year, args.league).snapshot(args.franchise)
    print(
        json.dumps(
            {
                "league": snapshot.league_name,
                "franchise_id": snapshot.franchise_id,
                "conference": snapshot.conference_name,
                "roster_size": snapshot.roster_size,
                "draft_rounds": snapshot.draft_rounds,
                "my_overall_picks": [pick.overall for pick in snapshot.my_picks],
                "drafted_in_conference": sorted(snapshot.drafted_player_ids),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

