#include<iostream>

#define Registry(Content) "modified"#Content

int main(){
    static int a=1;
    static float a=2.;
    return(0);
}