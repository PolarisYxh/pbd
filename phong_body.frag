varying vec3 fragNormal;
varying vec4 fragPosition;

uniform vec3 light_position;
uniform vec3 adjustNum;

void main() 
{ 
	vec3 V=normalize(vec3(fragPosition));
	vec3 N=normalize(fragNormal);
	vec3 L=normalize(light_position);
	vec3 L2=normalize(vec3(2.0,  1.0, 1.0));
	vec3 L3=normalize(vec3(2.0, -1.0, 1.0));

	L=normalize(vec3(-1, 0, 4));


	vec3 base_color=0;


	if(dot(N, L)>0)
		base_color=abs(dot(N, L))*0.9*vec3(222/255.0, 206/255.0, 201/255.0);
	else
		base_color=abs(dot(N, L))*0.9*vec3(0.74, 0.71, 0.64);




	gl_FragColor = vec4(base_color, 1);
		
	
 } 
