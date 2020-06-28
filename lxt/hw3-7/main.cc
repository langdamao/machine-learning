#include <cstring>
#include <iostream>
#include <cstdio>
#include <cmath>
using namespace std;
double u=0.0,v=0.0;
void getni(double ma[][2],double ni[][2]){
	double ban[2][2];
	ban[0][0]=ma[1][1];
	ban[1][1] = ma[0][0];
	ban[0][1] = -ma[0][1];
	ban[1][0] = -ma[1][0];
	double len = ma[0][0]*ma[1][1]-ma[0][1]*ma[1][0];
	for (int i=0;i<2;i++){
		for (int j=0;j<2;j++){
			ni[i][j] = ban[i][j]/len;
			//cout<<i<<' '<<j<<' '<<ma[i][j]<<' '<<ni[i][j]<<' '<<len<<endl;
		}
	}
}
void getdelta(double& tmpu,double& tmpv){
	double ma[2][2];
	ma[0][0] = exp(u)+v*v*exp(u*v)+2;
	ma[0][1] = exp(u*v)+u*v*exp(u*v)-2;
	ma[1][0] = ma[0][1];
	ma[1][1] = 4*exp(2*v)+u*u*exp(u*v)+4;
	double ni[2][2]; 
	getni(ma,ni);
	double a= exp(u)+v*exp(u*v)+2*u-2*v-3;
	double b= exp(2*v)*2+u*exp(u*v)-2*u+4*v-2;
	tmpu = -ni[0][0]*a-ni[0][1]*b;
	tmpv = -ni[1][0]*a-ni[1][1]*b;
}
int main(){
	for (int i=1;i<=5;i++){
		double tmpu = exp(u)+v*exp(u*v)+2*u-2*v-3;
		double tmpv = exp(2*v)*2+u*exp(u*v)-2*u+4*v-2;
		u = u- 0.01*tmpu;
		v = v-0.01*tmpv;
		double e = exp(u)+exp(2*v)+exp(u*v)+u*u-2*u*v+2*v*v-3*u-2*v;
		printf("%.6lf %.6lf %.6lf\n",u,v,e);
	}
	u = 0.0;v=0.0;
	for (int i=1;i<=5;i++){
		double tmpu,tmpv;
		getdelta(tmpu,tmpv);
		u = u+tmpu;
		v = v+tmpv;
//		u = u- 0.01*tmpu;
//		v = v-0.01*tmpv;
		double e = exp(u)+exp(2*v)+exp(u*v)+u*u-2*u*v+2*v*v-3*u-2*v;
		printf("%.6lf %.6lf %.6lf\n",u,v,e);
	}

	return 0;
}
