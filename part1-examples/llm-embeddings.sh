#!/bin/bash

SCRIPT_DIR=$(dirname "$0")
source "$SCRIPT_DIR/scripts/log.sh"

log_info "Crééons les embeddings pour 'France'..."
log_info "Attention, ça va faire beaucoup de contenu !"
c uvx llm embed -c 'France' -m 3-large

log_info "On enlève les embeddings précédents... (si il y en a)"
c uvx llm collections delete countries

log_info "On prend les pays à comparer et on calcule leurs embeddings"
c cat $SCRIPT_DIR/embed-countries.csv

log_info "5 secondes de pause pour vous laisser contempler le fichier..."
c sleep 5

log_info "On calcule les embeddings..."
c uvx llm embed-multi countries -m 3-large --store $SCRIPT_DIR/embed-countries.csv

log_info "Et maintenant on compare les embeddings... avec 'France'..."
c uvx llm similar countries -c "France"

log_info "Ou avec 'pays dont la capitale est Rome'..."
c uvx llm similar countries -c "pays dont la capitale est Rome"

log_info "Ou avec quelque chose qui n'a rien à voir..."
c uvx llm similar countries -c "yaourt bio"