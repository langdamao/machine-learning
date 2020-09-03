#include <cstring>
#include <cstdio>
#include <iostream>
using namespace std;
int z1(int x1,int x2){
	return x2*x2-2*x1+3;
}
int z2(int x1,int x2){
	return x1*x1-2*x2-3;
}
int main(){
	int x[7][3] = {
		1,0,-1,
		0,1,-1,
		0,-1,-1,
		-1,0,1,
		0,2,1,
		0,-2,1,
		-2,0,1
	};
	for (int i=0;i<7;i++){
		printf("%d %d %d\n",z1(x[i][0],x[i][1]),z2(x[i][0],x[i][1]),x[i][2]);
	}
	return 0;
}

