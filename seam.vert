attribute vec4 pos;

void main()
{
	gl_Position=gl_ModelViewProjectionMatrix *pos;
}