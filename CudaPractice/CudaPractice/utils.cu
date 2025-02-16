#include<iostream>
#include<vector>

#include "globals.cuh"
#include "vectork.cuh"


//utility functions
pair<vf, vf> parseInputs() {
	int n;
	vf a, b;
	cin >> n;
	for (int i = 0;i < n;i++) { float tmp; cin >> tmp; a.push_back(tmp); }
	for (int i = 0;i < n;i++) { float tmp; cin >> tmp; b.push_back(tmp); }

	return make_pair(a, b);
}

pair<mf, mf> parseInputsM() {
	int m, n;
	cin >> m; cin >> n;
	mf A(m, vf(n));
	for (int i = 0;i < m;i++) { for (int j = 0;j < n;j++) { float tmp; cin >> tmp; A[i][j] = tmp; } }

	cin >> m; cin >> n;
	mf B(m, vf(n));
	for (int i = 0;i < m;i++) { for (int j = 0;j < n;j++) { float tmp; cin >> tmp; B[i][j] = tmp; } }

	return make_pair(A, B);
}

void printMatrix(mf& M) {
	int m = (int)M.size(); int n = (int)M[0].size();
	for (int i = 0;i < m;i++) { for (int j = 0;j < n;j++) { cout << M[i][j] << " "; } cout << endl; }
}


void unitTest() {
	pair<mf, mf> x = parseInputsM();
	printMatrix(x.first); printMatrix(x.second);
	multMatrix(x.first, x.second);
}