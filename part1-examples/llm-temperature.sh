#!/bin/bash

SCRIPT_DIR=$(dirname "$0")
source "$SCRIPT_DIR/scripts/log.sh"

log_info "On va continuer la phrase 'Paris is...' avec une température de 0"
c uv run llm 'continue the following: "Paris is..."' -o temperature 0

log_info "Si on recommence, on obtient un résultat à peu près identique"
c uv run llm 'continue the following: "Paris is..."' -o temperature 0

log_info "Maintenant testons avec une température de 2"
c uv run llm 'continue the following: "Paris is..."' -o temperature 2 -o max_tokens 100

log_info "On a du mettre une limite parce que c'était trop chaotique."
log_info "On peut aussi jouer avec top_p pour s'assurer que le résultat est plus cohérent"
c uv run llm 'continue the following: "Paris is..."' -o temperature 2 -o top_p 0.999

