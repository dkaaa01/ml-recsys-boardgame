# Hybrid and terminal outputs

weight_validation.csv and weight_validation.png show validation NDCG@10 for the three ItemCF weights. Weight 0.25 has the highest score.

cli_guest.png and cli_profile.png display actual English console output saved in outputs/demo_guest.txt and outputs/demo_profile.txt. Both examples use 4 players and a 60-minute limit on the 200-user, 200-game sample.

Pressing Enter selects ranking by training rating count. Profile 0 selects saved hybrid scores and excludes the user's training-rated games. Final model metrics use the separate 20,000-user, 2,000-game experiment.

    .venv/bin/python recommend_cli.py --data-dir data/sample --scores outputs/scores.csv
