#pragma once
#ifndef UTILS_CUH
#define	UTILS_CUH

#include "globals.cuh"

vf addVector(vf& a, vf& b);
pair<vf, vf> parseInputs();
void unitTest();
#endif