varying vec3 fragNormal;
varying vec4 fragPosition;

uniform int adjustNum;
uniform vec3 light_position;


void main() 
{ 
	vec3 V=normalize(vec3(fragPosition));
	vec3 N=normalize(fragNormal);
	vec3 L=normalize(light_position);
	vec3 L2=normalize(vec3(2.0,  1.0, 1.0));
	vec3 L3=normalize(vec3(2.0, -1.0, 1.0));


	L=normalize(vec3(-1, 0, 4));


	vec3 base_color;
	vec3 adjustColor=vec3(0.0,0.0,0.0);

	if(adjustNum==0)
		adjustColor=vec3(139.0/255.0,192.0/255.0,184.0/255.0);//ÂÌ
		//adjustColor=vec3(255.0/255.0,131.0/255.0,131.0/255.0);
	else if(adjustNum==1)
		adjustColor=vec3(238.0/255.0,114.0/255.0,136.0/255.0);//ºì
	else if(adjustNum==2)
		adjustColor=vec3(218.0/255.0,182.0/255.0,167.0/255.0);//×Ø
	else if(adjustNum==3)
		adjustColor=vec3(249.0/255.0,227.0/255.0,158.0/255.0);//»Æ
	else if(adjustNum==4)
		adjustColor=vec3(73.0/255.0,75.0/255.0,65.0/255.0);  //¿¨ÆäÂÌ
	else if(adjustNum==5)
		adjustColor=vec3(141.0/255.0,110.0/255.0,86.0/255.0);//Êº»Æ
	else if(adjustNum==6)
		adjustColor=vec3(44.0/255.0,49.0/255.0,57.0/255.0); //À¶»Ò
	else if(adjustNum==7)
		adjustColor=vec3(25.0/255.0,31.0/255.0,52.0/255.0);  //ÉîÀ¶
	else if(adjustNum==8)
		adjustColor=vec3(44.0/255.0,148.0/255.0,209.0/255.0);
	else if(adjustNum==9)
		adjustColor=vec3(218.0/255.0,182.0/255.0,167.0/255.0);
	else if(adjustNum==10)
		adjustColor=vec3(157.0/255.0,156.0/255.0,158.0/255.0);
	else if(adjustNum==11)
		adjustColor=vec3(249.0/255.0,227.0/255.0,158.0/255.0);
	else if(adjustNum==12)
		adjustColor=vec3(73.0/255.0,75.0/255.0,65.0/255.0);
	else if(adjustNum==13)
		adjustColor=vec3(141.0/255.0,110.0/255.0,86.0/255.0);
	else if(adjustNum==14)
		adjustColor=vec3(44.0/255.0,49.0/255.0,57.0/255.0);
	else if(adjustNum==15)
		adjustColor=vec3(25.0/255.0,31.0/255.0,52.0/255.0);
	else if(adjustNum==16)
		adjustColor=vec3(44.0/255.0,148.0/255.0,209.0/255.0);
	else if(adjustNum==17)
		adjustColor=vec3(139.0/255.0,192.0/255.0,184.0/255.0);
	else if(adjustNum==18)
		adjustColor=vec3(238.0/255.0,114.0/255.0,136.0/255.0);
	else if(adjustNum==19)
		adjustColor=vec3(218.0/255.0,182.0/255.0,167.0/255.0);
	else if(adjustNum==20)
		adjustColor=vec3(249.0/255.0,227.0/255.0,158.0/255.0);
	else if(adjustNum==21)
		adjustColor=vec3(73.0/255.0,75.0/255.0,65.0/255.0);
	else if(adjustNum==22)
		adjustColor=vec3(141.0/255.0,110.0/255.0,86.0/255.0);
	else if(adjustNum==23)
		adjustColor=vec3(44.0/255.0,49.0/255.0,57.0/255.0);
	else if(adjustNum==24)
		adjustColor=vec3(25.0/255.0,31.0/255.0,52.0/255.0);
	else if(adjustNum==25)
		adjustColor=vec3(44.0/255.0,148.0/255.0,209.0/255.0);

	if(dot(N, L)>0)
		base_color=abs(dot(N, L))*adjustColor;
	else
		base_color=abs(dot(N, L))*vec3(0.6, 0.8, 1.0);




	gl_FragColor = vec4(base_color, 1);
		
	
 } 
