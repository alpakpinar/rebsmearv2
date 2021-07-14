#!/bin/bash

# Combine the ROOT trees in the specified output directory via hadd command.
INDIR=${1}

TARGETDIR=${INDIR}/combined
mkdir -p ${TARGETDIR}

pushd ${TARGETDIR}
hadd JetHT_rebalanced_combined.root ../JetHT_*rebalanced_tree*.root
popd
