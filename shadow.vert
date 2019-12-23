//attribute vec4 vertex_position;     //In the local space
//attribute vec4 vertex_normal;       //In the local space


varying vec4 fragment_position;

void main()
{ 

	fragment_position = gl_Vertex; 	//In the world space


	gl_Position = gl_ModelViewProjectionMatrix * gl_Vertex;   
}
