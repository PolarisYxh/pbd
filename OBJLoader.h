#ifndef __OBJLoader_h__
#define __OBJLoader_h__

#include <string>
#include <sstream>
#include <iostream>
#include <fstream>
#include <ctime>
#include <iomanip>
#include <vector>
#include <memory>
#include <array>

using namespace std;

namespace Utilities
{
	/** \brief Struct to store the position and normal indices
	*/
	struct StringTools
	{
	public:

		static void tokenize(const std::string& str, std::vector<std::string>& tokens, const std::string& delimiters = " ")
		{
			std::string::size_type lastPos = str.find_first_not_of(delimiters, 0);
			std::string::size_type pos = str.find_first_of(delimiters, lastPos);

			while (std::string::npos != pos || std::string::npos != lastPos)
			{
				tokens.push_back(str.substr(lastPos, pos - lastPos));
				lastPos = str.find_first_not_of(delimiters, pos);
				pos = str.find_first_of(delimiters, lastPos);
			}
		}

	};

	struct MeshFaceIndices
	{
		int posIndices[3];
		int texIndices[3];
		int normalIndices[3];
	};

	/** \brief Read for OBJ files.
	*/
	class OBJLoader
	{
	public:
		using Vec3f = std::array<float, 3>;
		using Vec2f = std::array<float, 2>;

		static void saveScaledAvatar(int curFrame)
		{
			
			string filename = "D:\\Program\\GPU_PBD_GARMENT_QT\\data\\walk\\sequence\\walk" + to_string(curFrame) + ".obj";
			std::ifstream filestream;
			filestream.open(filename.c_str());
			if (filestream.fail())
			{
				cout << "Failed to open file: " << filename << endl;
				system("pause");
				return;
			}
			string outFile= "D:\\Program\\GPU_PBD_GARMENT_QT\\data\\walk\\sequence-trans\\walk" + to_string(curFrame) + ".obj";
			std::ofstream filestream2;
			filestream2.open(outFile.c_str());
			if (filestream2.fail())
			{
				cout << "Failed to create file" << outFile << endl;
				system("pause");
				return;
			}

			std::string line_stream;
			std::vector<std::string> pos_buffer;
			std::vector<std::string> f_buffer;

			while (getline(filestream, line_stream))
			{
				std::stringstream str_stream(line_stream);
				std::string type_str;
				str_stream >> type_str;

				if (type_str == "v")
				{
					Vec3f pos;
					pos_buffer.clear();
					std::string parse_str = line_stream.substr(line_stream.find("v") + 1);
					StringTools::tokenize(parse_str, pos_buffer);
					vector<float> trans{ 0.045,1.105,0.754 };
					for (unsigned int i = 0; i < 3; i++)
						pos[i] = (stod(pos_buffer[i]) - trans[i]);

					filestream2 << "v " << pos.at(0) * 70 + 3.183 + 0.045 << " " << pos.at(1) * 70 + 94.166 + 1.105 << " " << pos.at(2) * 70 - 64.684 + 0.754 << "\n";
				}
				else
				{
					filestream2 << line_stream << "\n";
				}
			}
			filestream.close();
			filestream2.close();
		}

		/*  This funciton saves an OBJ file*/
		static void savePatternObj(const std::string& filename, std::vector<std::vector<Vec3f>> &xVec, std::vector<std::vector<MeshFaceIndices>>& facesVec,
			std::vector<std::vector<Vec3f>>& normalsVec, std::vector<std::vector<Vec2f>>& texcoordsVec)
		{
			std::ofstream filestream;
			filestream.open(filename.c_str());
			if (filestream.fail())
			{
				cout << "Failed to create file" << filename << endl;
				system("pause");
				return;
			}

			filestream << "#This file is save from current model\n \n";
			filestream << "mtllib " << "clothes" << ".mtl\n";
			for (int i = 0; i < xVec.size(); i++)
			{
				filestream << "g default\n";
				std::vector<Vec3f> xData = xVec.at(i);
				std::vector<Vec2f> texcoordsData = texcoordsVec.at(i);
				std::vector<Vec3f> normalsData = normalsVec.at(i);
				std::vector<MeshFaceIndices> facesData = facesVec.at(i);
				for (int j = 0; j < xData.size(); j++)
				{
					Vec3f x = xData.at(j);
					filestream << "v " << x.at(0)+0.045 << " " << x.at(1)+1.105 << " " << x.at(2)+0.754 << "\n";
				}
				for (int j = 0; j < texcoordsData.size(); j++)
				{
					Vec2f texcoords = texcoordsData.at(j);
					filestream << "vt " << texcoords.at(0) << " " << texcoords.at(1) << "\n";
				}
				for (int j = 0; j < normalsData.size(); j++)
				{
					Vec3f normals = normalsData.at(j);
					//filestream << "vn " << normals.at(0) << " " << normals.at(1) << " " << normals.at(2) << "\n";
				}
				filestream << "s off\n";
				filestream << "g clothes:" << "cloth" << i << "\n";
				filestream << "usemtl initialShadingGroup\n";
				for (int j = 0; j < facesData.size(); j++)
				{
					MeshFaceIndices faces = facesData.at(j);
					int* posIndices = faces.posIndices;
					int* texIndices = faces.texIndices;
					int* normalIndices = faces.normalIndices;
					filestream << "f " << posIndices[0] << "/" << texIndices[0] << "/" << normalIndices[0];
					filestream << " " << posIndices[1] << "/" << texIndices[1] << "/" << normalIndices[1];
					filestream << " " << posIndices[2] << "/" << texIndices[2] << "/" << normalIndices[2] << "\n";
				}
			}
			filestream.close();
			cout << "Saving file: " << filename << "       √" << endl;
		}

		/** This function loads an OBJ file.
		* Only triangulated meshes are supported.
		*/
		static void loadPatternObj(const std::string& filename, std::vector<std::vector<Vec3f>>* xVec, std::vector<std::vector<MeshFaceIndices>>* facesVec,
			std::vector<std::vector<Vec3f>>* normalsVec, std::vector<std::vector<Vec2f>>* texcoordsVec, const Vec3f& scale, unsigned int& vOffset, unsigned int& vtOffset, unsigned int& vnOffset)
		{
			cout << "Loading " << filename << endl;

			std::ifstream filestream;
			filestream.open(filename.c_str());
			if (filestream.fail())
			{
				cout << "Failed to open file: " << filename << endl;
				//system("pause");
				return;
			}

			std::string line_stream;
			bool vt = false;
			bool vn = false;

			std::vector<std::string> pos_buffer;
			std::vector<std::string> f_buffer;

			std::vector<Vec3f> x;
			std::vector<MeshFaceIndices> faces;
			faces.resize(0);
			std::vector<Vec3f> normals;
			std::vector<Vec2f> texcoords;

			std::string type_str, old_type_str;

			while (getline(filestream, line_stream))
			{
				if (line_stream == "g default") continue;
				std::stringstream str_stream(line_stream);
				str_stream >> type_str;  //得到空格前的数据

				if ((type_str == "v") && (old_type_str == "f"))
				{
					xVec->push_back(x);
					facesVec->push_back(faces);
					normalsVec->push_back(normals);
					texcoordsVec->push_back(texcoords);
					vOffset = vOffset + x.size();
					vtOffset = vtOffset + texcoords.size();
					vnOffset = vnOffset + normals.size();
					x.clear();
					faces.clear();
					normals.clear();
					texcoords.clear();
					vt = false;
					vn = false;
				}
				if ((type_str == "v"))
				{
					Vec3f pos;
					pos_buffer.clear();
					std::string parse_str = line_stream.substr(line_stream.find("v") + 1);  //去除v和空格后的字符串
					StringTools::tokenize(parse_str, pos_buffer);
					vector<float> trans{ 0.045,1.105,0.754 };
					for (unsigned int i = 0; i < 3; i++)
						pos[i] = (stod(pos_buffer[i])-trans[i]) * scale[i];
					x.push_back(pos);
				}
				else if (type_str == "vt")
				{
					Vec2f tex;
					pos_buffer.clear();
					std::string parse_str = line_stream.substr(line_stream.find("vt") + 2);
					StringTools::tokenize(parse_str, pos_buffer);
					for (unsigned int i = 0; i < 2; i++)
						tex[i] = stof(pos_buffer[i]);

					texcoords.push_back(tex);
					vt = true;

				}
				else if (type_str == "vn")
				{
					Vec3f nor;
					pos_buffer.clear();
					std::string parse_str = line_stream.substr(line_stream.find("vn") + 2);
					StringTools::tokenize(parse_str, pos_buffer);
					for (unsigned int i = 0; i < 3; i++)
						nor[i] = stof(pos_buffer[i]);

					normals.push_back(nor);
					vn = true;
				}
				else if (type_str == "f")
				{
					MeshFaceIndices faceIndex;
					if (vn && vt)
					{
						f_buffer.clear();
						std::string parse_str = line_stream.substr(line_stream.find("f") + 1);
						StringTools::tokenize(parse_str, f_buffer);
						for (int i = 0; i < 3; ++i)
						{
							pos_buffer.clear();
							StringTools::tokenize(f_buffer[i], pos_buffer, "/");
							faceIndex.posIndices[i] = stoi(pos_buffer[0]) - vOffset;
							faceIndex.texIndices[i] = stoi(pos_buffer[1]);
							faceIndex.normalIndices[i] = stoi(pos_buffer[2]);
						}
					}
					else if (vn)
					{
						f_buffer.clear();
						std::string parse_str = line_stream.substr(line_stream.find("f") + 1);
						StringTools::tokenize(parse_str, f_buffer);
						for (int i = 0; i < 3; ++i)
						{
							pos_buffer.clear();
							StringTools::tokenize(f_buffer[i], pos_buffer, "/");
							faceIndex.posIndices[i] = stoi(pos_buffer[0]);
							faceIndex.normalIndices[i] = stoi(pos_buffer[1]);
						}
					}
					else if (vt)
					{
						f_buffer.clear();
						std::string parse_str = line_stream.substr(line_stream.find("f") + 1);
						StringTools::tokenize(parse_str, f_buffer);
						for (int i = 0; i < 3; ++i)
						{
							pos_buffer.clear();
							StringTools::tokenize(f_buffer[i], pos_buffer, "/");
							faceIndex.posIndices[i] = stoi(pos_buffer[0]);
							faceIndex.texIndices[i] = stoi(pos_buffer[1]);
						}
					}
					else
					{
						f_buffer.clear();
						std::string parse_str = line_stream.substr(line_stream.find("f") + 1);
						StringTools::tokenize(parse_str, f_buffer);
						for (int i = 0; i < 3; ++i)
						{
							faceIndex.posIndices[i] = stoi(f_buffer[i]);
						}
					}
					faces.push_back(faceIndex);
				}

				old_type_str = type_str;
			}
			xVec->push_back(x);
			facesVec->push_back(faces);
			normalsVec->push_back(normals);
			texcoordsVec->push_back(texcoords);
			vOffset = vOffset + x.size();
			vtOffset = vtOffset + texcoords.size();
			vnOffset = vnOffset + normals.size();
			filestream.close();
			x.clear();
			faces.clear();
			normals.clear();
			texcoords.clear();
		}

		static void loadObjVertex(const std::string& filename, std::vector<Vec3f>* x)
		{
			cout << "Loading " << filename << endl;

			std::ifstream filestream;
			filestream.open(filename.c_str());
			if (filestream.fail())
			{
				cout << "Failed to open file: " << filename << endl;
				system("pause");
				return;
			}
			std::string line_stream;
			std::vector<std::string> pos_buffer;
			std::vector<std::string> f_buffer;

			while (getline(filestream, line_stream))
			{
				std::stringstream str_stream(line_stream);
				std::string type_str;
				str_stream >> type_str;

				if (type_str == "v")
				{
					Vec3f pos;
					pos_buffer.clear();
					std::string parse_str = line_stream.substr(line_stream.find("v") + 1);
					StringTools::tokenize(parse_str, pos_buffer);
					vector<float> trans{ 0.045,1.105,0.754 };
					for (unsigned int i = 0; i < 3; i++)
						pos[i] = (stod(pos_buffer[i]) - trans[i]);
					/*for (unsigned int i = 0; i < 3; i++)
						pos[i] = stof(pos_buffer[i]);*/

					x->push_back(pos);
				}
			}
			filestream.close();
		}

		static void loadObj(const std::string & filename, std::vector<Vec3f> * x, std::vector<MeshFaceIndices> * faces, std::vector<Vec3f> * normals, std::vector<Vec2f> * texcoords, const Vec3f & scale)
		{
			cout << "Loading " << filename << endl;

			std::ifstream filestream;
			filestream.open(filename.c_str());
			if (filestream.fail())
			{
				cout << "Failed to open file: " << filename << endl;
				system("pause");
				return;
			}

			std::string line_stream;
			bool vt = false;
			bool vn = false;

			std::vector<std::string> pos_buffer;
			std::vector<std::string> f_buffer;

			while (getline(filestream, line_stream))
			{
				std::stringstream str_stream(line_stream);
				std::string type_str;
				str_stream >> type_str;

				if (type_str == "v")
				{
					Vec3f pos;
					pos_buffer.clear();
					std::string parse_str = line_stream.substr(line_stream.find("v") + 1);
					StringTools::tokenize(parse_str, pos_buffer);
					vector<float> trans{ 0.045,1.105,0.754 };
					for (unsigned int i = 0; i < 3; i++)
						pos[i] = (stod(pos_buffer[i]) - trans[i]) * scale[i];
					/*for (unsigned int i = 0; i < 3; i++)
						pos[i] = stof(pos_buffer[i]) * scale[i];*/

					x->push_back(pos);
				}
				else if (type_str == "vt")
				{
					if (texcoords != nullptr)
					{
						Vec2f tex;
						pos_buffer.clear();
						std::string parse_str = line_stream.substr(line_stream.find("vt") + 2);
						StringTools::tokenize(parse_str, pos_buffer);
						for (unsigned int i = 0; i < 2; i++)
							tex[i] = stof(pos_buffer[i]);

						texcoords->push_back(tex);
						vt = true;
					}
				}
				else if (type_str == "vn")
				{
					if (normals != nullptr)
					{
						Vec3f nor;
						pos_buffer.clear();
						std::string parse_str = line_stream.substr(line_stream.find("vn") + 2);
						StringTools::tokenize(parse_str, pos_buffer);
						for (unsigned int i = 0; i < 3; i++)
							nor[i] = stof(pos_buffer[i]);

						normals->push_back(nor);
						vn = true;
					}
				}
				else if (type_str == "f")
				{
					MeshFaceIndices faceIndex;
					if (vn && vt)
					{
						f_buffer.clear();
						std::string parse_str = line_stream.substr(line_stream.find("f") + 1);
						StringTools::tokenize(parse_str, f_buffer);
						for (int i = 0; i < 3; ++i)
						{
							pos_buffer.clear();
							StringTools::tokenize(f_buffer[i], pos_buffer, "/");
							faceIndex.posIndices[i] = stoi(pos_buffer[0]);
							faceIndex.texIndices[i] = stoi(pos_buffer[1]);
							faceIndex.normalIndices[i] = stoi(pos_buffer[2]);
						}
					}
					else if (vn)
					{
						f_buffer.clear();
						std::string parse_str = line_stream.substr(line_stream.find("f") + 1);
						StringTools::tokenize(parse_str, f_buffer);
						for (int i = 0; i < 3; ++i)
						{
							pos_buffer.clear();
							StringTools::tokenize(f_buffer[i], pos_buffer, "/");
							faceIndex.posIndices[i] = stoi(pos_buffer[0]);
							faceIndex.normalIndices[i] = stoi(pos_buffer[1]);
						}
					}
					else if (vt)
					{
						f_buffer.clear();
						std::string parse_str = line_stream.substr(line_stream.find("f") + 1);
						StringTools::tokenize(parse_str, f_buffer);
						for (int i = 0; i < 3; ++i)
						{
							pos_buffer.clear();
							StringTools::tokenize(f_buffer[i], pos_buffer, "/");
							faceIndex.posIndices[i] = stoi(pos_buffer[0]);
							faceIndex.texIndices[i] = stoi(pos_buffer[1]);
						}
					}
					else
					{
						f_buffer.clear();
						std::string parse_str = line_stream.substr(line_stream.find("f") + 1);
						StringTools::tokenize(parse_str, f_buffer);
						for (int i = 0; i < 3; ++i)
						{
							faceIndex.posIndices[i] = stoi(f_buffer[i]);
						}
					}
					faces->push_back(faceIndex);
				}
			}
			filestream.close();
		}

	};
}

#endif