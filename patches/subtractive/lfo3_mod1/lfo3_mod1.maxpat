{
	"patcher" : 	{
		"fileversion" : 1,
		"appversion" : 		{
			"major" : 8,
			"minor" : 0,
			"revision" : 5,
			"architecture" : "x64",
			"modernui" : 1
		}
,
		"classnamespace" : "box",
		"rect" : [ -1566.0, -153.0, 1252.0, 504.0 ],
		"bglocked" : 0,
		"openinpresentation" : 1,
		"default_fontsize" : 12.0,
		"default_fontface" : 0,
		"default_fontname" : "Arial",
		"gridonopen" : 1,
		"gridsize" : [ 15.0, 15.0 ],
		"gridsnaponopen" : 1,
		"objectsnaponopen" : 1,
		"statusbarvisible" : 2,
		"toolbarvisible" : 1,
		"lefttoolbarpinned" : 0,
		"toptoolbarpinned" : 0,
		"righttoolbarpinned" : 0,
		"bottomtoolbarpinned" : 0,
		"toolbars_unpinned_last_save" : 0,
		"tallnewobj" : 0,
		"boxanimatetime" : 200,
		"enablehscroll" : 1,
		"enablevscroll" : 1,
		"devicewidth" : 0.0,
		"description" : "",
		"digest" : "",
		"tags" : "",
		"style" : "",
		"subpatcher_template" : "",
		"boxes" : [ 			{
				"box" : 				{
					"id" : "obj-145",
					"linecount" : 2,
					"maxclass" : "comment",
					"numinlets" : 1,
					"numoutlets" : 0,
					"patching_rect" : [ 1020.180013020833258, 413.0, 36.0, 33.0 ],
					"text" : "ENVEOR",
					"textjustification" : 1
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-144",
					"linecount" : 2,
					"maxclass" : "comment",
					"numinlets" : 1,
					"numoutlets" : 0,
					"patching_rect" : [ 983.180013020833258, 413.0, 36.0, 33.0 ],
					"text" : "ENVEOD",
					"textjustification" : 1
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-143",
					"linecount" : 2,
					"maxclass" : "comment",
					"numinlets" : 1,
					"numoutlets" : 0,
					"patching_rect" : [ 945.641682942708258, 413.0, 36.0, 33.0 ],
					"text" : "LFO\nEOC",
					"textjustification" : 1
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-123",
					"maxclass" : "button",
					"numinlets" : 1,
					"numoutlets" : 1,
					"outlettype" : [ "bang" ],
					"parameter_enable" : 0,
					"patching_rect" : [ 1026.180013020833258, 389.5, 24.0, 24.0 ]
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-124",
					"maxclass" : "button",
					"numinlets" : 1,
					"numoutlets" : 1,
					"outlettype" : [ "bang" ],
					"parameter_enable" : 0,
					"patching_rect" : [ 989.180013020833258, 389.5, 24.0, 24.0 ]
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-125",
					"maxclass" : "button",
					"numinlets" : 1,
					"numoutlets" : 1,
					"outlettype" : [ "bang" ],
					"parameter_enable" : 0,
					"patching_rect" : [ 952.180013020833258, 389.5, 24.0, 24.0 ]
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-108",
					"linecount" : 2,
					"maxclass" : "comment",
					"numinlets" : 1,
					"numoutlets" : 0,
					"patching_rect" : [ 1161.810953776041515, 47.0, 252.6412353515625, 33.0 ],
					"text" : "Note: the provided data directory must exist, and must contain an empty 'wavs' directory"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-109",
					"maxclass" : "message",
					"numinlets" : 2,
					"numoutlets" : 1,
					"outlettype" : [ "" ],
					"patching_rect" : [ 1090.263346354166515, 119.382080078125, 29.5, 22.0 ],
					"text" : "0"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-111",
					"maxclass" : "newobj",
					"numinlets" : 1,
					"numoutlets" : 1,
					"outlettype" : [ "signal" ],
					"patching_rect" : [ 1133.513346354166515, 119.382080078125, 69.0, 22.0 ],
					"text" : "sfrecord~ 1"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-112",
					"maxclass" : "message",
					"numinlets" : 2,
					"numoutlets" : 1,
					"outlettype" : [ "" ],
					"patching_rect" : [ 1065.513346354166515, 13.5, 79.0, 22.0 ],
					"text" : "resume 9000"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-113",
					"maxclass" : "message",
					"numinlets" : 2,
					"numoutlets" : 1,
					"outlettype" : [ "" ],
					"patching_rect" : [ 1011.103800455729129, 13.5, 31.0, 22.0 ],
					"text" : "stop"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-114",
					"maxclass" : "message",
					"numinlets" : 2,
					"numoutlets" : 1,
					"outlettype" : [ "" ],
					"patching_rect" : [ 962.013346354166629, 13.5, 32.0, 22.0 ],
					"text" : "start"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-117",
					"maxclass" : "message",
					"numinlets" : 2,
					"numoutlets" : 1,
					"outlettype" : [ "" ],
					"patching_rect" : [ 895.513346354166629, 13.5, 50.0, 22.0 ],
					"text" : "compile"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-118",
					"maxclass" : "newobj",
					"numinlets" : 1,
					"numoutlets" : 2,
					"outlettype" : [ "", "" ],
					"patching_rect" : [ 895.513346354166629, 52.5, 251.0, 22.0 ],
					"saved_object_attributes" : 					{
						"filename" : "generator_rand.js",
						"parameter_enable" : 0
					}
,
					"text" : "js generator_rand.js 4 10000 data/test 250 50"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-120",
					"maxclass" : "newobj",
					"numinlets" : 1,
					"numoutlets" : 1,
					"outlettype" : [ "" ],
					"patching_rect" : [ 895.513346354166629, 119.382080078125, 137.0, 22.0 ],
					"text" : "fromsymbol @separator"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-121",
					"maxclass" : "newobj",
					"numinlets" : 3,
					"numoutlets" : 3,
					"outlettype" : [ "", "", "" ],
					"patching_rect" : [ 895.513346354166629, 86.382080078125, 312.0, 22.0 ],
					"text" : "route /p /note"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-76",
					"maxclass" : "message",
					"numinlets" : 2,
					"numoutlets" : 1,
					"outlettype" : [ "" ],
					"patching_rect" : [ 511.2811279296875, 76.632080078125, 83.0, 22.0 ],
					"text" : "fcm_lstm_gen"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-79",
					"maxclass" : "message",
					"numinlets" : 2,
					"numoutlets" : 1,
					"outlettype" : [ "" ],
					"patching_rect" : [ 438.904541015625, 76.632080078125, 56.0, 22.0 ],
					"text" : "fcm_lstm"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-20",
					"maxclass" : "message",
					"numinlets" : 2,
					"numoutlets" : 1,
					"outlettype" : [ "" ],
					"patching_rect" : [ 511.2811279296875, 45.0660400390625, 81.0, 22.0 ],
					"text" : "fcm_cnn_gen"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-46",
					"maxclass" : "message",
					"numinlets" : 2,
					"numoutlets" : 1,
					"outlettype" : [ "" ],
					"patching_rect" : [ 438.904541015625, 45.0660400390625, 54.0, 22.0 ],
					"text" : "fcm_cnn"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-29",
					"maxclass" : "message",
					"numinlets" : 2,
					"numoutlets" : 1,
					"outlettype" : [ "" ],
					"patching_rect" : [ 511.2811279296875, 13.5, 81.0, 22.0 ],
					"text" : "fcm_dnn_gen"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-24",
					"maxclass" : "message",
					"numinlets" : 2,
					"numoutlets" : 1,
					"outlettype" : [ "" ],
					"patching_rect" : [ 438.904541015625, 13.5, 55.0, 22.0 ],
					"text" : "fcm_dnn"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-119",
					"linecount" : 2,
					"maxclass" : "message",
					"numinlets" : 2,
					"numoutlets" : 1,
					"outlettype" : [ "" ],
					"patching_rect" : [ 30.8466796875, 185.882080078125, 271.9334716796875, 35.0 ],
					"text" : "roundedrect 0.411264 0.554365 0.25 0.25 4 4, setcolor 1. 1. 1. 1., fill, repaint"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-116",
					"maxclass" : "newobj",
					"numinlets" : 1,
					"numoutlets" : 1,
					"outlettype" : [ "" ],
					"patching_rect" : [ 30.8466796875, 100.632080078125, 79.0, 22.0 ],
					"text" : "prepend load"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-115",
					"maxclass" : "newobj",
					"numinlets" : 1,
					"numoutlets" : 1,
					"outlettype" : [ "" ],
					"patching_rect" : [ 30.8466796875, 52.632080078125, 163.0, 22.0 ],
					"text" : "sprintf models/%s/timbremap"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-4",
					"maxclass" : "newobj",
					"numinlets" : 1,
					"numoutlets" : 1,
					"outlettype" : [ "" ],
					"patching_rect" : [ 30.8466796875, 76.632080078125, 58.0, 22.0 ],
					"text" : "jg.relpath"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-83",
					"maxclass" : "newobj",
					"numinlets" : 1,
					"numoutlets" : 1,
					"outlettype" : [ "signal" ],
					"patching_rect" : [ 1136.41796875, 233.5, 39.0, 22.0 ],
					"text" : "tanh~"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-107",
					"maxclass" : "newobj",
					"numinlets" : 0,
					"numoutlets" : 1,
					"outlettype" : [ "" ],
					"patching_rect" : [ 685.2303466796875, 434.382080078125, 43.0, 22.0 ],
					"text" : "r route"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-106",
					"maxclass" : "newobj",
					"numinlets" : 0,
					"numoutlets" : 1,
					"outlettype" : [ "" ],
					"patching_rect" : [ 627.5128173828125, 100.882080078125, 43.0, 22.0 ],
					"text" : "r route"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-105",
					"maxclass" : "newobj",
					"numinlets" : 1,
					"numoutlets" : 0,
					"patching_rect" : [ 257.7086181640625, 52.5, 45.0, 22.0 ],
					"text" : "s route"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-104",
					"maxclass" : "newobj",
					"numinlets" : 5,
					"numoutlets" : 1,
					"outlettype" : [ "" ],
					"patching_rect" : [ 739.5128173828125, 434.382080078125, 61.0, 22.0 ],
					"text" : "switch 4"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-103",
					"maxclass" : "message",
					"numinlets" : 2,
					"numoutlets" : 1,
					"outlettype" : [ "" ],
					"patching_rect" : [ 146.0, 13.5, 29.5, 22.0 ],
					"presentation" : 1,
					"presentation_rect" : [ 144.1533203125, 52.034576416015625, 29.5, 22.0 ],
					"text" : "am"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-99",
					"maxclass" : "message",
					"numinlets" : 2,
					"numoutlets" : 1,
					"outlettype" : [ "" ],
					"patching_rect" : [ 375.399658203125, 13.5, 29.5, 22.0 ],
					"text" : "4"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-98",
					"maxclass" : "message",
					"numinlets" : 2,
					"numoutlets" : 1,
					"outlettype" : [ "" ],
					"patching_rect" : [ 336.1693115234375, 13.5, 29.5, 22.0 ],
					"text" : "3"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-96",
					"maxclass" : "message",
					"numinlets" : 2,
					"numoutlets" : 1,
					"outlettype" : [ "" ],
					"patching_rect" : [ 296.93896484375, 13.5, 29.5, 22.0 ],
					"text" : "2"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-84",
					"maxclass" : "message",
					"numinlets" : 2,
					"numoutlets" : 1,
					"outlettype" : [ "" ],
					"patching_rect" : [ 257.7086181640625, 13.5, 29.5, 22.0 ],
					"text" : "1"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-45",
					"maxclass" : "newobj",
					"numinlets" : 2,
					"numoutlets" : 4,
					"outlettype" : [ "", "", "", "" ],
					"patching_rect" : [ 627.5128173828125, 139.882080078125, 50.5, 22.0 ],
					"text" : "gate 4"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-97",
					"linecount" : 5,
					"maxclass" : "comment",
					"numinlets" : 1,
					"numoutlets" : 0,
					"patching_rect" : [ 183.489501953125, 282.76416015625, 230.0, 74.0 ],
					"text" : "TO DO: \n- timbremapctrl sends updated toggles to timbremap, which load/switch models based on which parameters are under control"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-87",
					"maxclass" : "toggle",
					"numinlets" : 1,
					"numoutlets" : 1,
					"outlettype" : [ "int" ],
					"parameter_enable" : 0,
					"patching_rect" : [ 362.156158447265625, 348.0, 24.0, 24.0 ]
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-86",
					"maxclass" : "toggle",
					"numinlets" : 1,
					"numoutlets" : 1,
					"outlettype" : [ "int" ],
					"parameter_enable" : 0,
					"patching_rect" : [ 335.822845458984375, 348.0, 24.0, 24.0 ]
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-85",
					"maxclass" : "message",
					"numinlets" : 2,
					"numoutlets" : 1,
					"outlettype" : [ "" ],
					"patching_rect" : [ 154.989501953125, 458.76416015625, 255.0, 22.0 ],
					"text" : "param"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-82",
					"maxclass" : "toggle",
					"numinlets" : 1,
					"numoutlets" : 1,
					"outlettype" : [ "int" ],
					"parameter_enable" : 0,
					"patching_rect" : [ 309.489501953125, 348.0, 24.0, 24.0 ]
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-78",
					"maxclass" : "toggle",
					"numinlets" : 1,
					"numoutlets" : 1,
					"outlettype" : [ "int" ],
					"parameter_enable" : 0,
					"patching_rect" : [ 283.156158447265625, 348.0, 24.0, 24.0 ]
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-44",
					"maxclass" : "newobj",
					"numinlets" : 1,
					"numoutlets" : 0,
					"patching_rect" : [ 227.2801513671875, 409.882080078125, 25.0, 22.0 ],
					"text" : "s p"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-43",
					"maxclass" : "newobj",
					"numinlets" : 8,
					"numoutlets" : 5,
					"outlettype" : [ "", "", "", "", "" ],
					"patching_rect" : [ 269.989501953125, 409.882080078125, 111.166664123535156, 22.0 ],
					"saved_object_attributes" : 					{
						"filename" : "timbremapctrl",
						"parameter_enable" : 0
					}
,
					"text" : "js timbremapctrl 4"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-40",
					"maxclass" : "newobj",
					"numinlets" : 1,
					"numoutlets" : 1,
					"outlettype" : [ "bang" ],
					"patching_rect" : [ 739.5128173828125, 467.382080078125, 22.0, 22.0 ],
					"text" : "t b"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-38",
					"maxclass" : "newobj",
					"numinlets" : 1,
					"numoutlets" : 0,
					"patching_rect" : [ 459.2811279296875, 458.382080078125, 25.0, 22.0 ],
					"text" : "s p"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-28",
					"maxclass" : "newobj",
					"numinlets" : 1,
					"numoutlets" : 1,
					"outlettype" : [ "" ],
					"patching_rect" : [ 459.2811279296875, 434.382080078125, 90.0, 22.0 ],
					"text" : "prepend param"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-64",
					"maxclass" : "newobj",
					"numinlets" : 0,
					"numoutlets" : 1,
					"outlettype" : [ "" ],
					"patching_rect" : [ 283.7801513671875, 100.632080078125, 22.0, 22.0 ],
					"text" : "r s"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-75",
					"maxclass" : "newobj",
					"numinlets" : 0,
					"numoutlets" : 1,
					"outlettype" : [ "" ],
					"patching_rect" : [ 216.8060302734375, 100.632080078125, 19.0, 22.0 ],
					"text" : "r t"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-163",
					"maxclass" : "message",
					"numinlets" : 2,
					"numoutlets" : 1,
					"outlettype" : [ "" ],
					"patching_rect" : [ 65.8466796875, 13.632080078125, 34.0, 22.0 ],
					"presentation" : 1,
					"presentation_rect" : [ 64.0, 52.166656494140625, 34.0, 22.0 ],
					"text" : "pwm"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-162",
					"maxclass" : "message",
					"numinlets" : 2,
					"numoutlets" : 1,
					"outlettype" : [ "" ],
					"patching_rect" : [ 106.0966796875, 13.632080078125, 29.5, 22.0 ],
					"presentation" : 1,
					"presentation_rect" : [ 104.25, 52.166656494140625, 29.5, 22.0 ],
					"text" : "fcm"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-161",
					"maxclass" : "message",
					"numinlets" : 2,
					"numoutlets" : 1,
					"outlettype" : [ "" ],
					"patching_rect" : [ 30.8466796875, 13.632080078125, 29.5, 22.0 ],
					"presentation" : 1,
					"presentation_rect" : [ 29.0, 52.166656494140625, 29.5, 22.0 ],
					"text" : "fm"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-146",
					"maxclass" : "newobj",
					"numinlets" : 0,
					"numoutlets" : 1,
					"outlettype" : [ "" ],
					"patching_rect" : [ 121.846649169921875, 100.632080078125, 23.0, 22.0 ],
					"text" : "r p"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-147",
					"maxclass" : "newobj",
					"numinlets" : 0,
					"numoutlets" : 1,
					"outlettype" : [ "" ],
					"patching_rect" : [ 148.3466796875, 100.632080078125, 23.0, 22.0 ],
					"text" : "r b"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-77",
					"maxclass" : "newobj",
					"numinlets" : 3,
					"numoutlets" : 3,
					"outlettype" : [ "list", "list", "list" ],
					"patching_rect" : [ 30.8466796875, 139.882080078125, 271.9334716796875, 22.0 ],
					"text" : "jg.timbremap 3 4"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-138",
					"maxclass" : "newobj",
					"numinlets" : 1,
					"numoutlets" : 1,
					"outlettype" : [ "" ],
					"patching_rect" : [ 176.7801513671875, 250.382080078125, 126.0, 22.0 ],
					"text" : "prepend showtracking"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-139",
					"maxclass" : "toggle",
					"numinlets" : 1,
					"numoutlets" : 1,
					"outlettype" : [ "int" ],
					"parameter_enable" : 0,
					"patching_rect" : [ 176.7801513671875, 222.382080078125, 24.0, 24.0 ]
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-48",
					"maxclass" : "newobj",
					"numinlets" : 1,
					"numoutlets" : 0,
					"patching_rect" : [ 30.8466796875, 458.76416015625, 25.0, 22.0 ],
					"text" : "s b"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-135",
					"maxclass" : "newobj",
					"numinlets" : 1,
					"numoutlets" : 1,
					"outlettype" : [ "bang" ],
					"patching_rect" : [ 45.1373291015625, 225.882080078125, 58.0, 22.0 ],
					"text" : "loadbang"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-137",
					"maxclass" : "newobj",
					"numinlets" : 1,
					"numoutlets" : 1,
					"outlettype" : [ "" ],
					"patching_rect" : [ 30.8466796875, 434.76416015625, 84.0, 22.0 ],
					"text" : "prepend block"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-57",
					"maxclass" : "message",
					"numinlets" : 2,
					"numoutlets" : 1,
					"outlettype" : [ "" ],
					"patching_rect" : [ 45.1373291015625, 250.382080078125, 125.0, 22.0 ],
					"text" : "elementcolor 0. 0. 0."
				}

			}
, 			{
				"box" : 				{
					"bgcolor" : [ 0.0, 0.0, 0.0, 1.0 ],
					"elementcolor" : [ 0.0, 0.0, 0.0, 1.0 ],
					"id" : "obj-140",
					"maxclass" : "blocks.multitouch",
					"numinlets" : 1,
					"numoutlets" : 1,
					"outlettype" : [ "int" ],
					"patching_rect" : [ 30.8466796875, 285.26416015625, 135.0, 135.0 ],
					"presentation" : 1,
					"presentation_rect" : [ 34.0, 99.0, 135.0, 135.0 ],
					"showtracking" : 0
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-47",
					"maxclass" : "newobj",
					"numinlets" : 4,
					"numoutlets" : 1,
					"outlettype" : [ "" ],
					"patching_rect" : [ 459.2811279296875, 409.882080078125, 63.0, 22.0 ],
					"text" : "pack f f f f"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-42",
					"maxclass" : "newobj",
					"numinlets" : 1,
					"numoutlets" : 4,
					"outlettype" : [ "float", "float", "float", "float" ],
					"patching_rect" : [ 459.2811279296875, 139.882080078125, 74.0, 22.0 ],
					"text" : "unpack f f f f"
				}

			}
, 			{
				"box" : 				{
					"embedstate" : [ [ "separator", "" ], [ "prefix", "" ] ],
					"id" : "obj-49",
					"linecount" : 2,
					"maxclass" : "newobj",
					"numinlets" : 22,
					"numoutlets" : 1,
					"outlettype" : [ "" ],
					"patching_rect" : [ 24.8466796875, 729.882080078125, 801.399658203125, 35.0 ],
					"saved_object_attributes" : 					{
						"filename" : "jg.ms",
						"parameter_enable" : 0
					}
,
					"text" : "js jg.ms /lfo/rate /lfo/duty /lfo/shape /mod/fm /mod/pwm /mod/fcm /mod/am /env/atk /env/dec /env/sus /env/rel /env/shape /env/vs /env/rt /dco/coarse /dco/fine /dco/duty /dco/wave /dcf/fc /dcf/q /dcf/ms /dcf/type"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-37",
					"maxclass" : "newobj",
					"numinlets" : 2,
					"numoutlets" : 1,
					"outlettype" : [ "" ],
					"patching_rect" : [ 895.513346354166629, 273.5, 54.0, 22.0 ],
					"text" : "pack 0 0"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-30",
					"maxclass" : "newobj",
					"numinlets" : 1,
					"numoutlets" : 1,
					"outlettype" : [ "" ],
					"patching_rect" : [ 895.513346354166629, 312.5, 101.0, 22.0 ],
					"text" : "prepend midinote"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-27",
					"maxclass" : "newobj",
					"numinlets" : 1,
					"numoutlets" : 7,
					"outlettype" : [ "signal", "signal", "", "", "", "", "" ],
					"patching_rect" : [ 895.513346354166629, 350.5, 189.0, 22.0 ],
					"text" : "poly~ jg.pvoice.mod1~ 6 @steal 1"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-19",
					"maxclass" : "newobj",
					"numinlets" : 1,
					"numoutlets" : 1,
					"outlettype" : [ "" ],
					"patching_rect" : [ 1011.103800455729129, 234.5, 70.0, 22.0 ],
					"text" : "loadmess 1"
				}

			}
, 			{
				"box" : 				{
					"fontface" : 0,
					"id" : "obj-2",
					"linecount" : 2,
					"maxclass" : "comment",
					"numinlets" : 1,
					"numoutlets" : 0,
					"patching_rect" : [ 489.7086181640625, 209.382080078125, 62.0, 33.0 ],
					"presentation" : 1,
					"presentation_linecount" : 2,
					"presentation_rect" : [ 264.85369873046875, 66.666656494140625, 62.0, 33.0 ],
					"text" : "Duty Cycle",
					"textcolor" : [ 0.996078431606293, 0.996078431606293, 0.996078431606293, 1.0 ],
					"textjustification" : 1
				}

			}
, 			{
				"box" : 				{
					"fontface" : 0,
					"id" : "obj-1",
					"maxclass" : "comment",
					"numinlets" : 1,
					"numoutlets" : 0,
					"patching_rect" : [ 542.2916259765625, 215.882080078125, 58.0, 20.0 ],
					"presentation" : 1,
					"presentation_rect" : [ 316.86419677734375, 73.166656494140625, 58.0, 20.0 ],
					"text" : "Shape",
					"textcolor" : [ 0.996078431606293, 0.996078431606293, 0.996078431606293, 1.0 ],
					"textjustification" : 1
				}

			}
, 			{
				"box" : 				{
					"floatoutput" : 1,
					"id" : "obj-6",
					"knobcolor" : [ 0.666666666666667, 0.72156862745098, 0.737254901960784, 1.0 ],
					"maxclass" : "slider",
					"numinlets" : 1,
					"numoutlets" : 1,
					"outlettype" : [ "" ],
					"parameter_enable" : 0,
					"patching_rect" : [ 561.2811279296875, 242.882080078125, 20.0, 140.0 ],
					"presentation" : 1,
					"presentation_rect" : [ 335.85369873046875, 100.166656494140625, 20.0, 140.0 ],
					"size" : 127.0
				}

			}
, 			{
				"box" : 				{
					"fontface" : 0,
					"id" : "obj-15",
					"maxclass" : "comment",
					"numinlets" : 1,
					"numoutlets" : 0,
					"patching_rect" : [ 443.09283447265625, 215.882080078125, 52.376571655273438, 20.0 ],
					"presentation" : 1,
					"presentation_rect" : [ 223.85369873046875, 73.166656494140625, 42.0, 20.0 ],
					"text" : "Rate",
					"textcolor" : [ 0.996078431606293, 0.996078431606293, 0.996078431606293, 1.0 ],
					"textjustification" : 1
				}

			}
, 			{
				"box" : 				{
					"fontface" : 1,
					"fontsize" : 16.0,
					"id" : "obj-17",
					"maxclass" : "comment",
					"numinlets" : 1,
					"numoutlets" : 0,
					"patching_rect" : [ 438.904541015625, 185.882080078125, 163.608261108398438, 24.0 ],
					"presentation" : 1,
					"presentation_rect" : [ 213.47711181640625, 43.166656494140625, 163.608261108398438, 24.0 ],
					"text" : "LFO",
					"textcolor" : [ 0.996078431606293, 0.996078431606293, 0.996078431606293, 1.0 ],
					"textjustification" : 1
				}

			}
, 			{
				"box" : 				{
					"floatoutput" : 1,
					"id" : "obj-21",
					"knobcolor" : [ 0.666666666666667, 0.72156862745098, 0.737254901960784, 1.0 ],
					"maxclass" : "slider",
					"numinlets" : 1,
					"numoutlets" : 1,
					"outlettype" : [ "" ],
					"parameter_enable" : 0,
					"patching_rect" : [ 511.2811279296875, 242.882080078125, 20.0, 140.0 ],
					"presentation" : 1,
					"presentation_rect" : [ 285.85369873046875, 100.166656494140625, 20.0, 140.0 ],
					"size" : 127.0
				}

			}
, 			{
				"box" : 				{
					"floatoutput" : 1,
					"id" : "obj-22",
					"knobcolor" : [ 0.666666666666667, 0.72156862745098, 0.737254901960784, 1.0 ],
					"maxclass" : "slider",
					"numinlets" : 1,
					"numoutlets" : 1,
					"outlettype" : [ "" ],
					"parameter_enable" : 0,
					"patching_rect" : [ 459.2811279296875, 242.882080078125, 20.0, 140.0 ],
					"presentation" : 1,
					"presentation_rect" : [ 233.85369873046875, 100.166656494140625, 20.0, 140.0 ],
					"size" : 127.0
				}

			}
, 			{
				"box" : 				{
					"fontface" : 1,
					"fontsize" : 16.0,
					"id" : "obj-3",
					"maxclass" : "comment",
					"numinlets" : 1,
					"numoutlets" : 0,
					"patching_rect" : [ 604.5128173828125, 185.882080078125, 216.7071533203125, 24.0 ],
					"presentation" : 1,
					"presentation_rect" : [ 379.085372924804688, 43.166656494140625, 216.7071533203125, 24.0 ],
					"text" : "MOD",
					"textcolor" : [ 0.996078431606293, 0.996078431606293, 0.996078431606293, 1.0 ],
					"textjustification" : 1
				}

			}
, 			{
				"box" : 				{
					"fontface" : 0,
					"id" : "obj-5",
					"maxclass" : "comment",
					"numinlets" : 1,
					"numoutlets" : 0,
					"patching_rect" : [ 760.5128173828125, 215.882080078125, 60.7071533203125, 20.0 ],
					"presentation" : 1,
					"presentation_rect" : [ 535.085372924804688, 73.166656494140625, 60.7071533203125, 20.0 ],
					"text" : "AM",
					"textcolor" : [ 0.996078431606293, 0.996078431606293, 0.996078431606293, 1.0 ],
					"textjustification" : 1
				}

			}
, 			{
				"box" : 				{
					"floatoutput" : 1,
					"id" : "obj-7",
					"knobcolor" : [ 0.666666666666667, 0.72156862745098, 0.737254901960784, 1.0 ],
					"maxclass" : "slider",
					"numinlets" : 1,
					"numoutlets" : 1,
					"outlettype" : [ "" ],
					"parameter_enable" : 0,
					"patching_rect" : [ 781.5128173828125, 242.882080078125, 20.0, 140.0 ],
					"presentation" : 1,
					"presentation_rect" : [ 556.085372924804688, 100.166656494140625, 20.0, 140.0 ],
					"size" : 127.0
				}

			}
, 			{
				"box" : 				{
					"fontface" : 0,
					"id" : "obj-9",
					"maxclass" : "comment",
					"numinlets" : 1,
					"numoutlets" : 0,
					"patching_rect" : [ 657.5128173828125, 215.882080078125, 62.0, 20.0 ],
					"presentation" : 1,
					"presentation_rect" : [ 432.085372924804688, 73.166656494140625, 62.0, 20.0 ],
					"text" : "PWM",
					"textcolor" : [ 0.996078431606293, 0.996078431606293, 0.996078431606293, 1.0 ],
					"textjustification" : 1
				}

			}
, 			{
				"box" : 				{
					"fontface" : 0,
					"id" : "obj-101",
					"maxclass" : "comment",
					"numinlets" : 1,
					"numoutlets" : 0,
					"patching_rect" : [ 708.5128173828125, 215.882080078125, 62.0, 20.0 ],
					"presentation" : 1,
					"presentation_rect" : [ 483.085372924804688, 73.166656494140625, 62.0, 20.0 ],
					"text" : "FCM",
					"textcolor" : [ 0.996078431606293, 0.996078431606293, 0.996078431606293, 1.0 ],
					"textjustification" : 1
				}

			}
, 			{
				"box" : 				{
					"floatoutput" : 1,
					"id" : "obj-10",
					"knobcolor" : [ 0.666666666666667, 0.72156862745098, 0.737254901960784, 1.0 ],
					"maxclass" : "slider",
					"numinlets" : 1,
					"numoutlets" : 1,
					"outlettype" : [ "" ],
					"parameter_enable" : 0,
					"patching_rect" : [ 729.5128173828125, 242.882080078125, 20.0, 140.0 ],
					"presentation" : 1,
					"presentation_rect" : [ 504.085372924804688, 100.166656494140625, 20.0, 140.0 ],
					"size" : 127.0
				}

			}
, 			{
				"box" : 				{
					"fontface" : 0,
					"id" : "obj-11",
					"maxclass" : "comment",
					"numinlets" : 1,
					"numoutlets" : 0,
					"patching_rect" : [ 616.5128173828125, 215.882080078125, 42.0, 20.0 ],
					"presentation" : 1,
					"presentation_rect" : [ 391.085372924804688, 73.166656494140625, 42.0, 20.0 ],
					"text" : "FM",
					"textcolor" : [ 0.996078431606293, 0.996078431606293, 0.996078431606293, 1.0 ],
					"textjustification" : 1
				}

			}
, 			{
				"box" : 				{
					"floatoutput" : 1,
					"id" : "obj-13",
					"knobcolor" : [ 0.666666666666667, 0.72156862745098, 0.737254901960784, 1.0 ],
					"maxclass" : "slider",
					"numinlets" : 1,
					"numoutlets" : 1,
					"outlettype" : [ "" ],
					"parameter_enable" : 0,
					"patching_rect" : [ 678.5128173828125, 242.882080078125, 20.0, 140.0 ],
					"presentation" : 1,
					"presentation_rect" : [ 453.085372924804688, 100.166656494140625, 20.0, 140.0 ],
					"size" : 127.0
				}

			}
, 			{
				"box" : 				{
					"floatoutput" : 1,
					"id" : "obj-14",
					"knobcolor" : [ 0.666666666666667, 0.72156862745098, 0.737254901960784, 1.0 ],
					"maxclass" : "slider",
					"numinlets" : 1,
					"numoutlets" : 1,
					"outlettype" : [ "" ],
					"parameter_enable" : 0,
					"patching_rect" : [ 627.5128173828125, 242.882080078125, 20.0, 140.0 ],
					"presentation" : 1,
					"presentation_rect" : [ 402.085372924804688, 100.166656494140625, 20.0, 140.0 ],
					"size" : 127.0
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-26",
					"maxclass" : "newobj",
					"numinlets" : 1,
					"numoutlets" : 3,
					"outlettype" : [ "int", "int", "int" ],
					"patching_rect" : [ 895.513346354166629, 234.5, 54.0, 22.0 ],
					"text" : "notein"
				}

			}
, 			{
				"box" : 				{
					"disabled" : [ 0 ],
					"id" : "obj-181",
					"itemtype" : 1,
					"maxclass" : "radiogroup",
					"numinlets" : 1,
					"numoutlets" : 1,
					"outlettype" : [ "" ],
					"parameter_enable" : 0,
					"patching_rect" : [ 398.7752685546875, 567.882080078125, 18.0, 18.0 ],
					"presentation" : 1,
					"presentation_rect" : [ 402.10003662109375, 321.0, 18.0, 18.0 ],
					"size" : 1,
					"values" : [ 0 ]
				}

			}
, 			{
				"box" : 				{
					"fontface" : 0,
					"id" : "obj-165",
					"linecount" : 2,
					"maxclass" : "comment",
					"numinlets" : 1,
					"numoutlets" : 0,
					"patching_rect" : [ 416.634033203125, 560.382080078125, 40.651683807373047, 33.0 ],
					"presentation" : 1,
					"presentation_linecount" : 2,
					"presentation_rect" : [ 421.95880126953125, 313.5, 40.651683807373047, 33.0 ],
					"text" : "MIDI Scale",
					"textcolor" : [ 0.996078431606293, 0.996078431606293, 0.996078431606293, 1.0 ],
					"textjustification" : 1
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-74",
					"maxclass" : "comment",
					"numinlets" : 1,
					"numoutlets" : 0,
					"patching_rect" : [ 416.7752685546875, 661.882080078125, 27.0, 20.0 ],
					"presentation" : 1,
					"presentation_rect" : [ 420.10003662109375, 415.0, 27.0, 20.0 ],
					"text" : "BP",
					"textcolor" : [ 0.996078431606293, 0.996078431606293, 0.996078431606293, 1.0 ]
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-88",
					"maxclass" : "comment",
					"numinlets" : 1,
					"numoutlets" : 0,
					"patching_rect" : [ 416.7752685546875, 640.882080078125, 27.0, 20.0 ],
					"presentation" : 1,
					"presentation_rect" : [ 420.10003662109375, 394.0, 27.0, 20.0 ],
					"text" : "HP",
					"textcolor" : [ 0.996078431606293, 0.996078431606293, 0.996078431606293, 1.0 ]
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-89",
					"maxclass" : "comment",
					"numinlets" : 1,
					"numoutlets" : 0,
					"patching_rect" : [ 416.7752685546875, 620.882080078125, 27.0, 20.0 ],
					"presentation" : 1,
					"presentation_rect" : [ 420.10003662109375, 374.0, 27.0, 20.0 ],
					"text" : "LP",
					"textcolor" : [ 0.996078431606293, 0.996078431606293, 0.996078431606293, 1.0 ]
				}

			}
, 			{
				"box" : 				{
					"disabled" : [ 0, 0, 0 ],
					"id" : "obj-90",
					"itemtype" : 0,
					"maxclass" : "radiogroup",
					"numinlets" : 1,
					"numoutlets" : 1,
					"outlettype" : [ "" ],
					"parameter_enable" : 0,
					"patching_rect" : [ 398.7752685546875, 625.882080078125, 18.0, 50.0 ],
					"presentation" : 1,
					"presentation_rect" : [ 402.10003662109375, 379.0, 18.0, 50.0 ],
					"size" : 3,
					"value" : 0
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-91",
					"linecount" : 2,
					"maxclass" : "comment",
					"numinlets" : 1,
					"numoutlets" : 0,
					"patching_rect" : [ 288.489501953125, 532.382080078125, 45.0, 33.0 ],
					"presentation" : 1,
					"presentation_linecount" : 2,
					"presentation_rect" : [ 291.81427001953125, 285.5, 45.0, 33.0 ],
					"text" : "Cutoff Freq",
					"textcolor" : [ 0.996078431606293, 0.996078431606293, 0.996078431606293, 1.0 ],
					"textjustification" : 1
				}

			}
, 			{
				"box" : 				{
					"floatoutput" : 1,
					"id" : "obj-92",
					"knobcolor" : [ 0.666666666666667, 0.72156862745098, 0.737254901960784, 1.0 ],
					"maxclass" : "slider",
					"numinlets" : 1,
					"numoutlets" : 1,
					"outlettype" : [ "" ],
					"parameter_enable" : 0,
					"patching_rect" : [ 300.989501953125, 565.882080078125, 20.0, 140.0 ],
					"presentation" : 1,
					"presentation_rect" : [ 304.31427001953125, 319.0, 20.0, 140.0 ],
					"size" : 127.0
				}

			}
, 			{
				"box" : 				{
					"fontface" : 0,
					"id" : "obj-93",
					"maxclass" : "comment",
					"numinlets" : 1,
					"numoutlets" : 0,
					"patching_rect" : [ 340.989501953125, 538.882080078125, 44.0, 20.0 ],
					"presentation" : 1,
					"presentation_rect" : [ 344.31427001953125, 292.0, 44.0, 20.0 ],
					"text" : "Q",
					"textcolor" : [ 0.996078431606293, 0.996078431606293, 0.996078431606293, 1.0 ],
					"textjustification" : 1
				}

			}
, 			{
				"box" : 				{
					"floatoutput" : 1,
					"id" : "obj-94",
					"knobcolor" : [ 0.666666666666667, 0.72156862745098, 0.737254901960784, 1.0 ],
					"maxclass" : "slider",
					"numinlets" : 1,
					"numoutlets" : 1,
					"outlettype" : [ "" ],
					"parameter_enable" : 0,
					"patching_rect" : [ 352.989501953125, 565.882080078125, 20.0, 140.0 ],
					"presentation" : 1,
					"presentation_rect" : [ 356.31427001953125, 319.0, 20.0, 140.0 ],
					"size" : 127.0
				}

			}
, 			{
				"box" : 				{
					"fontface" : 1,
					"fontsize" : 16.0,
					"id" : "obj-95",
					"maxclass" : "comment",
					"numinlets" : 1,
					"numoutlets" : 0,
					"patching_rect" : [ 278.989501953125, 508.882080078125, 184.7857666015625, 24.0 ],
					"presentation" : 1,
					"presentation_rect" : [ 282.31427001953125, 262.0, 184.7857666015625, 24.0 ],
					"text" : "DCF",
					"textcolor" : [ 0.996078431606293, 0.996078431606293, 0.996078431606293, 1.0 ],
					"textjustification" : 1
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-18",
					"maxclass" : "comment",
					"numinlets" : 1,
					"numoutlets" : 0,
					"patching_rect" : [ 215.0560302734375, 640.882080078125, 42.0, 20.0 ],
					"presentation" : 1,
					"presentation_rect" : [ 219.74871826171875, 394.0, 42.0, 20.0 ],
					"text" : "Pulse",
					"textcolor" : [ 0.996078431606293, 0.996078431606293, 0.996078431606293, 1.0 ]
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-59",
					"maxclass" : "comment",
					"numinlets" : 1,
					"numoutlets" : 0,
					"patching_rect" : [ 215.0560302734375, 619.882080078125, 58.0, 20.0 ],
					"presentation" : 1,
					"presentation_rect" : [ 219.74871826171875, 373.0, 58.0, 20.0 ],
					"text" : "Tri/Ramp",
					"textcolor" : [ 0.996078431606293, 0.996078431606293, 0.996078431606293, 1.0 ]
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-60",
					"maxclass" : "comment",
					"numinlets" : 1,
					"numoutlets" : 0,
					"patching_rect" : [ 215.0560302734375, 599.882080078125, 33.0, 20.0 ],
					"presentation" : 1,
					"presentation_rect" : [ 219.74871826171875, 353.0, 33.0, 20.0 ],
					"text" : "Sine",
					"textcolor" : [ 0.996078431606293, 0.996078431606293, 0.996078431606293, 1.0 ]
				}

			}
, 			{
				"box" : 				{
					"disabled" : [ 0, 0, 0 ],
					"id" : "obj-61",
					"itemtype" : 0,
					"maxclass" : "radiogroup",
					"numinlets" : 1,
					"numoutlets" : 1,
					"outlettype" : [ "" ],
					"parameter_enable" : 0,
					"patching_rect" : [ 197.0560302734375, 604.882080078125, 18.0, 50.0 ],
					"presentation" : 1,
					"presentation_rect" : [ 201.74871826171875, 358.0, 18.0, 50.0 ],
					"size" : 3,
					"value" : 2
				}

			}
, 			{
				"box" : 				{
					"fontface" : 1,
					"fontsize" : 16.0,
					"id" : "obj-41",
					"maxclass" : "comment",
					"numinlets" : 1,
					"numoutlets" : 0,
					"patching_rect" : [ 24.8466796875, 508.882080078125, 251.142822265625, 24.0 ],
					"presentation" : 1,
					"presentation_rect" : [ 29.5394287109375, 262.0, 251.142822265625, 24.0 ],
					"text" : "DCO",
					"textcolor" : [ 0.996078431606293, 0.996078431606293, 0.996078431606293, 1.0 ],
					"textjustification" : 1
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-33",
					"linecount" : 2,
					"maxclass" : "comment",
					"numinlets" : 1,
					"numoutlets" : 0,
					"patching_rect" : [ 89.3466796875, 532.382080078125, 34.0, 33.0 ],
					"presentation" : 1,
					"presentation_linecount" : 2,
					"presentation_rect" : [ 94.0394287109375, 285.5, 34.0, 33.0 ],
					"text" : "Fine\nFreq",
					"textcolor" : [ 0.996078431606293, 0.996078431606293, 0.996078431606293, 1.0 ],
					"textjustification" : 1
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-66",
					"maxclass" : "comment",
					"numinlets" : 1,
					"numoutlets" : 0,
					"patching_rect" : [ 30.8466796875, 538.882080078125, 49.0, 20.0 ],
					"presentation" : 1,
					"presentation_rect" : [ 35.5394287109375, 292.0, 49.0, 20.0 ],
					"text" : "Octave",
					"textcolor" : [ 0.996078431606293, 0.996078431606293, 0.996078431606293, 1.0 ],
					"textjustification" : 1
				}

			}
, 			{
				"box" : 				{
					"floatoutput" : 1,
					"id" : "obj-50",
					"knobcolor" : [ 0.666666666666667, 0.72156862745098, 0.737254901960784, 1.0 ],
					"maxclass" : "slider",
					"numinlets" : 1,
					"numoutlets" : 1,
					"outlettype" : [ "" ],
					"parameter_enable" : 0,
					"patching_rect" : [ 96.3466796875, 565.882080078125, 20.0, 140.0 ],
					"presentation" : 1,
					"presentation_rect" : [ 101.0394287109375, 319.0, 20.0, 140.0 ],
					"size" : 127.0
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-67",
					"knobcolor" : [ 0.666666666666667, 0.72156862745098, 0.737254901960784, 1.0 ],
					"maxclass" : "slider",
					"mult" : 24.0,
					"numinlets" : 1,
					"numoutlets" : 1,
					"outlettype" : [ "" ],
					"parameter_enable" : 0,
					"patching_rect" : [ 45.3466796875, 565.882080078125, 20.0, 140.0 ],
					"presentation" : 1,
					"presentation_rect" : [ 50.0394287109375, 319.0, 20.0, 140.0 ],
					"size" : 6.0
				}

			}
, 			{
				"box" : 				{
					"fontface" : 0,
					"id" : "obj-36",
					"linecount" : 2,
					"maxclass" : "comment",
					"numinlets" : 1,
					"numoutlets" : 0,
					"patching_rect" : [ 136.3466796875, 532.382080078125, 44.0, 33.0 ],
					"presentation" : 1,
					"presentation_linecount" : 2,
					"presentation_rect" : [ 141.0394287109375, 285.5, 44.0, 33.0 ],
					"text" : "Duty\nCycle",
					"textcolor" : [ 0.996078431606293, 0.996078431606293, 0.996078431606293, 1.0 ],
					"textjustification" : 1
				}

			}
, 			{
				"box" : 				{
					"floatoutput" : 1,
					"id" : "obj-69",
					"knobcolor" : [ 0.666666666666667, 0.72156862745098, 0.737254901960784, 1.0 ],
					"maxclass" : "slider",
					"numinlets" : 1,
					"numoutlets" : 1,
					"outlettype" : [ "" ],
					"parameter_enable" : 0,
					"patching_rect" : [ 148.3466796875, 565.882080078125, 20.0, 140.0 ],
					"presentation" : 1,
					"presentation_rect" : [ 153.0394287109375, 319.0, 20.0, 140.0 ],
					"size" : 127.0
				}

			}
, 			{
				"box" : 				{
					"fontface" : 0,
					"id" : "obj-34",
					"maxclass" : "comment",
					"numinlets" : 1,
					"numoutlets" : 0,
					"patching_rect" : [ 672.5787353515625, 539.1236572265625, 62.0, 20.0 ],
					"presentation" : 1,
					"presentation_rect" : [ 673.7071533203125, 292.2415771484375, 62.0, 20.0 ],
					"text" : "Shape",
					"textcolor" : [ 0.996078431606293, 0.996078431606293, 0.996078431606293, 1.0 ],
					"textjustification" : 1
				}

			}
, 			{
				"box" : 				{
					"floatoutput" : 1,
					"id" : "obj-35",
					"knobcolor" : [ 0.666666666666667, 0.72156862745098, 0.737254901960784, 1.0 ],
					"maxclass" : "slider",
					"numinlets" : 1,
					"numoutlets" : 1,
					"outlettype" : [ "" ],
					"parameter_enable" : 0,
					"patching_rect" : [ 693.5787353515625, 565.882080078125, 20.0, 140.0 ],
					"presentation" : 1,
					"presentation_rect" : [ 694.7071533203125, 319.0, 20.0, 140.0 ],
					"size" : 127.0
				}

			}
, 			{
				"box" : 				{
					"fontface" : 0,
					"id" : "obj-51",
					"maxclass" : "comment",
					"numinlets" : 1,
					"numoutlets" : 0,
					"patching_rect" : [ 762.219970703125, 651.882080078125, 53.0, 20.0 ],
					"presentation" : 1,
					"presentation_rect" : [ 763.348388671875, 405.0, 53.0, 20.0 ],
					"text" : "Retrig",
					"textcolor" : [ 0.996078431606293, 0.996078431606293, 0.996078431606293, 1.0 ],
					"textjustification" : 1
				}

			}
, 			{
				"box" : 				{
					"fontface" : 0,
					"id" : "obj-52",
					"linecount" : 2,
					"maxclass" : "comment",
					"numinlets" : 1,
					"numoutlets" : 0,
					"patching_rect" : [ 763.219970703125, 593.382080078125, 53.0, 33.0 ],
					"presentation" : 1,
					"presentation_linecount" : 2,
					"presentation_rect" : [ 764.348388671875, 346.5, 53.0, 33.0 ],
					"text" : "Velocity Scale",
					"textcolor" : [ 0.996078431606293, 0.996078431606293, 0.996078431606293, 1.0 ],
					"textjustification" : 1
				}

			}
, 			{
				"box" : 				{
					"disabled" : [ 0 ],
					"id" : "obj-53",
					"itemtype" : 1,
					"maxclass" : "radiogroup",
					"numinlets" : 1,
					"numoutlets" : 1,
					"outlettype" : [ "" ],
					"parameter_enable" : 0,
					"patching_rect" : [ 745.5787353515625, 653.882080078125, 18.0, 18.0 ],
					"presentation" : 1,
					"presentation_rect" : [ 746.7071533203125, 407.0, 18.0, 18.0 ],
					"size" : 1,
					"values" : [ 0 ]
				}

			}
, 			{
				"box" : 				{
					"disabled" : [ 0 ],
					"id" : "obj-54",
					"itemtype" : 1,
					"maxclass" : "radiogroup",
					"numinlets" : 1,
					"numoutlets" : 1,
					"outlettype" : [ "" ],
					"parameter_enable" : 0,
					"patching_rect" : [ 745.5787353515625, 600.882080078125, 18.0, 18.0 ],
					"presentation" : 1,
					"presentation_rect" : [ 746.7071533203125, 354.0, 18.0, 18.0 ],
					"size" : 1,
					"values" : [ 0 ]
				}

			}
, 			{
				"box" : 				{
					"fontface" : 1,
					"fontsize" : 16.0,
					"id" : "obj-81",
					"maxclass" : "comment",
					"numinlets" : 1,
					"numoutlets" : 0,
					"patching_rect" : [ 466.87158203125, 508.882080078125, 354.348388671875, 24.0 ],
					"presentation" : 1,
					"presentation_rect" : [ 468.0, 262.0, 354.348388671875, 24.0 ],
					"text" : "ENV",
					"textcolor" : [ 0.996078431606293, 0.996078431606293, 0.996078431606293, 1.0 ],
					"textjustification" : 1
				}

			}
, 			{
				"box" : 				{
					"fontface" : 0,
					"id" : "obj-72",
					"maxclass" : "comment",
					"numinlets" : 1,
					"numoutlets" : 0,
					"patching_rect" : [ 621.5787353515625, 539.1236572265625, 62.0, 20.0 ],
					"presentation" : 1,
					"presentation_rect" : [ 622.7071533203125, 292.2415771484375, 62.0, 20.0 ],
					"text" : "Release",
					"textcolor" : [ 0.996078431606293, 0.996078431606293, 0.996078431606293, 1.0 ],
					"textjustification" : 1
				}

			}
, 			{
				"box" : 				{
					"floatoutput" : 1,
					"id" : "obj-73",
					"knobcolor" : [ 0.666666666666667, 0.72156862745098, 0.737254901960784, 1.0 ],
					"maxclass" : "slider",
					"numinlets" : 1,
					"numoutlets" : 1,
					"outlettype" : [ "" ],
					"parameter_enable" : 0,
					"patching_rect" : [ 642.2303466796875, 565.882080078125, 20.0, 140.0 ],
					"presentation" : 1,
					"presentation_rect" : [ 643.3587646484375, 319.0, 20.0, 140.0 ],
					"size" : 127.0
				}

			}
, 			{
				"box" : 				{
					"fontface" : 0,
					"id" : "obj-55",
					"maxclass" : "comment",
					"numinlets" : 1,
					"numoutlets" : 0,
					"patching_rect" : [ 519.87158203125, 538.882080078125, 62.0, 20.0 ],
					"presentation" : 1,
					"presentation_rect" : [ 521.0, 292.0, 62.0, 20.0 ],
					"text" : "Decay",
					"textcolor" : [ 0.996078431606293, 0.996078431606293, 0.996078431606293, 1.0 ],
					"textjustification" : 1
				}

			}
, 			{
				"box" : 				{
					"fontface" : 0,
					"id" : "obj-63",
					"maxclass" : "comment",
					"numinlets" : 1,
					"numoutlets" : 0,
					"patching_rect" : [ 570.2303466796875, 539.1236572265625, 62.0, 20.0 ],
					"presentation" : 1,
					"presentation_rect" : [ 571.3587646484375, 292.2415771484375, 62.0, 20.0 ],
					"text" : "Sustain",
					"textcolor" : [ 0.996078431606293, 0.996078431606293, 0.996078431606293, 1.0 ],
					"textjustification" : 1
				}

			}
, 			{
				"box" : 				{
					"floatoutput" : 1,
					"id" : "obj-56",
					"knobcolor" : [ 0.666666666666667, 0.72156862745098, 0.737254901960784, 1.0 ],
					"maxclass" : "slider",
					"numinlets" : 1,
					"numoutlets" : 1,
					"outlettype" : [ "" ],
					"parameter_enable" : 0,
					"patching_rect" : [ 590.882080078125, 565.882080078125, 20.0, 140.0 ],
					"presentation" : 1,
					"presentation_rect" : [ 592.010498046875, 319.0, 20.0, 140.0 ],
					"size" : 127.0
				}

			}
, 			{
				"box" : 				{
					"fontface" : 0,
					"id" : "obj-65",
					"maxclass" : "comment",
					"numinlets" : 1,
					"numoutlets" : 0,
					"patching_rect" : [ 476.87158203125, 538.882080078125, 42.0, 20.0 ],
					"presentation" : 1,
					"presentation_rect" : [ 478.0, 292.0, 42.0, 20.0 ],
					"text" : "Attack",
					"textcolor" : [ 0.996078431606293, 0.996078431606293, 0.996078431606293, 1.0 ],
					"textjustification" : 1
				}

			}
, 			{
				"box" : 				{
					"floatoutput" : 1,
					"id" : "obj-68",
					"knobcolor" : [ 0.666666666666667, 0.72156862745098, 0.737254901960784, 1.0 ],
					"maxclass" : "slider",
					"numinlets" : 1,
					"numoutlets" : 1,
					"outlettype" : [ "" ],
					"parameter_enable" : 0,
					"patching_rect" : [ 540.87158203125, 565.882080078125, 20.0, 140.0 ],
					"presentation" : 1,
					"presentation_rect" : [ 542.0, 319.0, 20.0, 140.0 ],
					"size" : 127.0
				}

			}
, 			{
				"box" : 				{
					"floatoutput" : 1,
					"id" : "obj-71",
					"knobcolor" : [ 0.666666666666667, 0.72156862745098, 0.737254901960784, 1.0 ],
					"maxclass" : "slider",
					"numinlets" : 1,
					"numoutlets" : 1,
					"outlettype" : [ "" ],
					"parameter_enable" : 0,
					"patching_rect" : [ 487.87158203125, 565.882080078125, 20.0, 140.0 ],
					"presentation" : 1,
					"presentation_rect" : [ 489.0, 319.0, 20.0, 140.0 ],
					"size" : 127.0
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-23",
					"maxclass" : "message",
					"numinlets" : 2,
					"numoutlets" : 1,
					"outlettype" : [ "" ],
					"patching_rect" : [ 206.3060302734375, 543.382080078125, 29.5, 22.0 ],
					"presentation" : 1,
					"presentation_rect" : [ 208.27398681640625, 296.5, 29.5, 22.0 ],
					"text" : "63."
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-62",
					"maxclass" : "preset",
					"numinlets" : 1,
					"numoutlets" : 4,
					"outlettype" : [ "preset", "int", "preset", "int" ],
					"patching_rect" : [ 1011.103800455729129, 275.0, 69.999999999999886, 30.0 ],
					"presentation" : 1,
					"presentation_rect" : [ 632.802530924479015, 191.666656494140625, 56.904622395833485, 28.0 ],
					"preset_data" : [ 						{
							"number" : 1,
							"data" : [ 5, "obj-8", "live.gain~", "float", 0.049309410154819, 5, "obj-71", "slider", "float", 12.610740661621094, 5, "obj-68", "slider", "float", 12.484685897827148, 5, "obj-56", "slider", "float", 102.708259582519531, 5, "obj-73", "slider", "float", 36.923519134521484, 5, "obj-54", "radiogroup", "list", 0, 5, "obj-53", "radiogroup", "list", 0, 5, "obj-35", "slider", "float", 126.023078918457031, 5, "obj-69", "slider", "float", 63.0, 5, "obj-67", "slider", "float", 2.0, 5, "obj-50", "slider", "float", 63.0, 5, "obj-61", "radiogroup", "int", 2, 5, "obj-94", "slider", "float", 78.883941650390625, 5, "obj-92", "slider", "float", 47.496349334716797, 5, "obj-90", "radiogroup", "int", 0, 5, "obj-181", "radiogroup", "list", 0, 5, "obj-14", "slider", "float", 0.0, 5, "obj-13", "slider", "float", 0.0, 5, "obj-10", "slider", "float", 0.0, 5, "obj-7", "slider", "float", 0.0, 5, "obj-22", "slider", "float", 51.091503143310547, 5, "obj-21", "slider", "float", 53.061103820800781, 5, "obj-6", "slider", "float", 62.278846740722656, 5, "obj-139", "toggle", "int", 0, 5, "obj-78", "toggle", "int", 0, 5, "obj-82", "toggle", "int", 0, 5, "obj-86", "toggle", "int", 0, 5, "obj-87", "toggle", "int", 0 ]
						}
 ]
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-8",
					"lastchannelcount" : 0,
					"maxclass" : "live.gain~",
					"numinlets" : 2,
					"numoutlets" : 5,
					"outlettype" : [ "signal", "signal", "", "float", "list" ],
					"parameter_enable" : 1,
					"patching_rect" : [ 1136.41796875, 272.5, 46.0, 100.0 ],
					"presentation" : 1,
					"presentation_rect" : [ 718.7071533203125, 66.666656494140625, 46.0, 100.0 ],
					"saved_attribute_attributes" : 					{
						"valueof" : 						{
							"parameter_initial" : [ 0.0 ],
							"parameter_shortname" : "live.gain~",
							"parameter_type" : 0,
							"parameter_unitstyle" : 4,
							"parameter_mmin" : -70.0,
							"parameter_longname" : "live.gain~",
							"parameter_mmax" : 6.0
						}

					}
,
					"varname" : "live.gain~"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-12",
					"maxclass" : "ezdac~",
					"numinlets" : 2,
					"numoutlets" : 0,
					"patching_rect" : [ 1136.41796875, 389.0, 45.0, 45.0 ],
					"presentation" : 1,
					"presentation_rect" : [ 718.7071533203125, 183.166656494140625, 45.0, 45.0 ]
				}

			}
, 			{
				"box" : 				{
					"fontface" : 1,
					"id" : "obj-32",
					"maxclass" : "comment",
					"numinlets" : 1,
					"numoutlets" : 0,
					"patching_rect" : [ 843.91796875, 556.5, 77.0, 20.0 ],
					"presentation" : 1,
					"presentation_rect" : [ 829.91796875, 283.2415771484375, 77.0, 20.0 ],
					"text" : "LFO",
					"textcolor" : [ 0.996078431606293, 0.95686274766922, 0.95686274766922, 1.0 ],
					"textjustification" : 1
				}

			}
, 			{
				"box" : 				{
					"automatic" : 1,
					"bufsize" : 32,
					"drawstyle" : 1,
					"id" : "obj-39",
					"maxclass" : "scope~",
					"numinlets" : 2,
					"numoutlets" : 0,
					"patching_rect" : [ 923.8466796875, 516.5, 300.0, 100.0 ],
					"presentation" : 1,
					"presentation_rect" : [ 909.8466796875, 243.2415771484375, 300.0, 100.0 ]
				}

			}
, 			{
				"box" : 				{
					"fontface" : 1,
					"id" : "obj-128",
					"linecount" : 2,
					"maxclass" : "comment",
					"numinlets" : 1,
					"numoutlets" : 0,
					"patching_rect" : [ 843.91796875, 667.0, 77.0, 33.0 ],
					"presentation" : 1,
					"presentation_linecount" : 2,
					"presentation_rect" : [ 829.91796875, 393.7415771484375, 77.0, 33.0 ],
					"text" : "Output\nSignal",
					"textcolor" : [ 0.996078431606293, 0.95686274766922, 0.95686274766922, 1.0 ],
					"textjustification" : 1
				}

			}
, 			{
				"box" : 				{
					"automatic" : 1,
					"bufsize" : 32,
					"drawstyle" : 1,
					"id" : "obj-129",
					"maxclass" : "scope~",
					"numinlets" : 2,
					"numoutlets" : 0,
					"patching_rect" : [ 923.8466796875, 633.5, 300.0, 100.0 ],
					"presentation" : 1,
					"presentation_rect" : [ 909.8466796875, 360.2415771484375, 300.0, 100.0 ]
				}

			}
, 			{
				"box" : 				{
					"angle" : 270.0,
					"background" : 1,
					"bgcolor" : [ 0.396078431372549, 0.396078431372549, 0.396078431372549, 0.84 ],
					"id" : "obj-130",
					"maxclass" : "panel",
					"mode" : 0,
					"numinlets" : 1,
					"numoutlets" : 0,
					"patching_rect" : [ 838.3466796875, 501.882080078125, 399.5, 243.117919921875 ],
					"presentation" : 1,
					"presentation_rect" : [ 824.3466796875, 228.6236572265625, 399.5, 243.117919921875 ],
					"proportion" : 0.5
				}

			}
, 			{
				"box" : 				{
					"angle" : 270.0,
					"background" : 1,
					"bgcolor" : [ 0.498039215686275, 0.498039215686275, 0.498039215686275, 1.0 ],
					"id" : "obj-80",
					"maxclass" : "panel",
					"mode" : 0,
					"numinlets" : 1,
					"numoutlets" : 0,
					"patching_rect" : [ 466.87158203125, 501.882080078125, 354.348388671875, 216.0 ],
					"presentation" : 1,
					"presentation_rect" : [ 468.0, 255.0, 354.348388671875, 216.0 ],
					"proportion" : 0.5
				}

			}
, 			{
				"box" : 				{
					"angle" : 270.0,
					"background" : 1,
					"bgcolor" : [ 0.498039215686275, 0.498039215686275, 0.498039215686275, 1.0 ],
					"id" : "obj-25",
					"maxclass" : "panel",
					"mode" : 0,
					"numinlets" : 1,
					"numoutlets" : 0,
					"patching_rect" : [ 438.904541015625, 178.882080078125, 163.608261108398438, 216.0 ],
					"presentation" : 1,
					"presentation_rect" : [ 213.47711181640625, 36.166656494140625, 163.608261108398438, 216.0 ],
					"proportion" : 0.5
				}

			}
, 			{
				"box" : 				{
					"angle" : 270.0,
					"background" : 1,
					"bgcolor" : [ 0.498039215686275, 0.498039215686275, 0.498039215686275, 1.0 ],
					"id" : "obj-70",
					"maxclass" : "panel",
					"mode" : 0,
					"numinlets" : 1,
					"numoutlets" : 0,
					"patching_rect" : [ 24.8466796875, 501.882080078125, 251.142822265625, 216.0 ],
					"presentation" : 1,
					"presentation_rect" : [ 29.5394287109375, 255.0, 251.142822265625, 216.0 ],
					"proportion" : 0.5
				}

			}
, 			{
				"box" : 				{
					"angle" : 270.0,
					"background" : 1,
					"bgcolor" : [ 0.498039215686275, 0.498039215686275, 0.498039215686275, 1.0 ],
					"id" : "obj-16",
					"maxclass" : "panel",
					"mode" : 0,
					"numinlets" : 1,
					"numoutlets" : 0,
					"patching_rect" : [ 604.5128173828125, 178.882080078125, 216.7071533203125, 216.0 ],
					"presentation" : 1,
					"presentation_rect" : [ 379.085372924804688, 36.166656494140625, 216.7071533203125, 216.0 ],
					"proportion" : 0.5
				}

			}
, 			{
				"box" : 				{
					"background" : 1,
					"id" : "obj-58",
					"maxclass" : "blocks.pad",
					"numinlets" : 1,
					"numoutlets" : 3,
					"outlettype" : [ "int", "dictionary", "int" ],
					"patching_rect" : [ 25.88238525390625, 282.76416015625, 144.9285888671875, 140.0 ],
					"presentation" : 1,
					"presentation_rect" : [ 29.0, 94.0, 145.333343505859375, 145.166656494140625 ]
				}

			}
, 			{
				"box" : 				{
					"angle" : 270.0,
					"background" : 1,
					"bgcolor" : [ 0.498039215686275, 0.498039215686275, 0.498039215686275, 1.0 ],
					"id" : "obj-100",
					"maxclass" : "panel",
					"mode" : 0,
					"numinlets" : 1,
					"numoutlets" : 0,
					"patching_rect" : [ 278.989501953125, 501.882080078125, 184.7857666015625, 216.0 ],
					"presentation" : 1,
					"presentation_rect" : [ 282.31427001953125, 255.0, 184.7857666015625, 216.0 ],
					"proportion" : 0.5
				}

			}
 ],
		"lines" : [ 			{
				"patchline" : 				{
					"destination" : [ "obj-104", 3 ],
					"order" : 0,
					"source" : [ "obj-10", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-49", 5 ],
					"order" : 1,
					"source" : [ "obj-10", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-115", 0 ],
					"order" : 1,
					"source" : [ "obj-103", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-99", 0 ],
					"order" : 0,
					"source" : [ "obj-103", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-40", 0 ],
					"order" : 0,
					"source" : [ "obj-104", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-47", 3 ],
					"order" : 1,
					"source" : [ "obj-104", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-45", 0 ],
					"source" : [ "obj-106", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-104", 0 ],
					"source" : [ "obj-107", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-111", 0 ],
					"source" : [ "obj-109", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-118", 0 ],
					"source" : [ "obj-112", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-109", 0 ],
					"order" : 0,
					"source" : [ "obj-113", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-118", 0 ],
					"order" : 1,
					"source" : [ "obj-113", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-118", 0 ],
					"source" : [ "obj-114", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-4", 0 ],
					"source" : [ "obj-115", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-77", 0 ],
					"source" : [ "obj-116", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-118", 0 ],
					"source" : [ "obj-117", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-111", 0 ],
					"source" : [ "obj-118", 1 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-121", 0 ],
					"source" : [ "obj-118", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-140", 0 ],
					"source" : [ "obj-119", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-42", 0 ],
					"source" : [ "obj-120", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-120", 0 ],
					"source" : [ "obj-121", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-30", 0 ],
					"source" : [ "obj-121", 1 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-104", 2 ],
					"order" : 0,
					"source" : [ "obj-13", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-49", 4 ],
					"order" : 1,
					"source" : [ "obj-13", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-57", 0 ],
					"source" : [ "obj-135", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-48", 0 ],
					"source" : [ "obj-137", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-140", 0 ],
					"source" : [ "obj-138", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-138", 0 ],
					"source" : [ "obj-139", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-104", 1 ],
					"order" : 0,
					"source" : [ "obj-14", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-43", 6 ],
					"order" : 1,
					"source" : [ "obj-14", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-49", 3 ],
					"order" : 2,
					"source" : [ "obj-14", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-137", 0 ],
					"source" : [ "obj-140", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-77", 0 ],
					"source" : [ "obj-146", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-77", 0 ],
					"source" : [ "obj-147", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-115", 0 ],
					"order" : 1,
					"source" : [ "obj-161", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-84", 0 ],
					"order" : 0,
					"source" : [ "obj-161", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-115", 0 ],
					"order" : 1,
					"source" : [ "obj-162", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-98", 0 ],
					"order" : 0,
					"source" : [ "obj-162", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-115", 0 ],
					"order" : 1,
					"source" : [ "obj-163", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-96", 0 ],
					"order" : 0,
					"source" : [ "obj-163", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-49", 20 ],
					"source" : [ "obj-181", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-62", 0 ],
					"source" : [ "obj-19", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-115", 0 ],
					"order" : 1,
					"source" : [ "obj-20", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-98", 0 ],
					"order" : 0,
					"source" : [ "obj-20", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-40", 0 ],
					"order" : 0,
					"source" : [ "obj-21", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-43", 2 ],
					"order" : 2,
					"source" : [ "obj-21", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-47", 1 ],
					"order" : 1,
					"source" : [ "obj-21", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-49", 1 ],
					"order" : 3,
					"source" : [ "obj-21", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-43", 0 ],
					"order" : 1,
					"source" : [ "obj-22", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-47", 0 ],
					"order" : 0,
					"source" : [ "obj-22", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-49", 0 ],
					"order" : 2,
					"source" : [ "obj-22", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-50", 0 ],
					"order" : 1,
					"source" : [ "obj-23", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-69", 0 ],
					"order" : 0,
					"source" : [ "obj-23", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-115", 0 ],
					"order" : 1,
					"source" : [ "obj-24", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-98", 0 ],
					"order" : 0,
					"source" : [ "obj-24", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-37", 1 ],
					"source" : [ "obj-26", 1 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-37", 0 ],
					"source" : [ "obj-26", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-123", 0 ],
					"source" : [ "obj-27", 4 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-124", 0 ],
					"source" : [ "obj-27", 3 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-125", 0 ],
					"source" : [ "obj-27", 2 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-129", 0 ],
					"order" : 1,
					"source" : [ "obj-27", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-39", 0 ],
					"source" : [ "obj-27", 1 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-83", 0 ],
					"order" : 0,
					"source" : [ "obj-27", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-38", 0 ],
					"source" : [ "obj-28", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-115", 0 ],
					"order" : 1,
					"source" : [ "obj-29", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-98", 0 ],
					"order" : 0,
					"source" : [ "obj-29", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-27", 0 ],
					"source" : [ "obj-30", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-49", 11 ],
					"source" : [ "obj-35", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-30", 0 ],
					"source" : [ "obj-37", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-116", 0 ],
					"source" : [ "obj-4", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-47", 0 ],
					"source" : [ "obj-40", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-21", 0 ],
					"source" : [ "obj-42", 1 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-22", 0 ],
					"source" : [ "obj-42", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-45", 1 ],
					"source" : [ "obj-42", 3 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-6", 0 ],
					"source" : [ "obj-42", 2 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-85", 1 ],
					"source" : [ "obj-43", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-10", 0 ],
					"source" : [ "obj-45", 2 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-13", 0 ],
					"source" : [ "obj-45", 1 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-14", 0 ],
					"source" : [ "obj-45", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-7", 0 ],
					"source" : [ "obj-45", 3 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-115", 0 ],
					"order" : 1,
					"source" : [ "obj-46", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-98", 0 ],
					"order" : 0,
					"source" : [ "obj-46", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-28", 0 ],
					"source" : [ "obj-47", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-49", 15 ],
					"source" : [ "obj-50", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-49", 13 ],
					"source" : [ "obj-53", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-49", 12 ],
					"source" : [ "obj-54", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-49", 9 ],
					"source" : [ "obj-56", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-140", 0 ],
					"source" : [ "obj-57", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-40", 0 ],
					"order" : 0,
					"source" : [ "obj-6", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-43", 4 ],
					"order" : 2,
					"source" : [ "obj-6", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-47", 2 ],
					"order" : 1,
					"source" : [ "obj-6", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-49", 2 ],
					"order" : 3,
					"source" : [ "obj-6", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-49", 17 ],
					"source" : [ "obj-61", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-77", 2 ],
					"source" : [ "obj-64", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-49", 14 ],
					"source" : [ "obj-67", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-49", 8 ],
					"source" : [ "obj-68", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-49", 16 ],
					"source" : [ "obj-69", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-104", 4 ],
					"order" : 0,
					"source" : [ "obj-7", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-49", 6 ],
					"order" : 1,
					"source" : [ "obj-7", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-49", 7 ],
					"source" : [ "obj-71", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-49", 10 ],
					"source" : [ "obj-73", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-77", 1 ],
					"source" : [ "obj-75", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-115", 0 ],
					"order" : 1,
					"source" : [ "obj-76", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-98", 0 ],
					"order" : 0,
					"source" : [ "obj-76", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-119", 1 ],
					"order" : 0,
					"source" : [ "obj-77", 2 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-119", 0 ],
					"order" : 1,
					"source" : [ "obj-77", 2 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-42", 0 ],
					"source" : [ "obj-77", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-43", 1 ],
					"source" : [ "obj-78", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-115", 0 ],
					"order" : 1,
					"source" : [ "obj-79", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-98", 0 ],
					"order" : 0,
					"source" : [ "obj-79", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-12", 1 ],
					"source" : [ "obj-8", 1 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-12", 0 ],
					"source" : [ "obj-8", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-43", 3 ],
					"source" : [ "obj-82", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-111", 0 ],
					"order" : 2,
					"source" : [ "obj-83", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-8", 1 ],
					"order" : 0,
					"source" : [ "obj-83", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-8", 0 ],
					"order" : 1,
					"source" : [ "obj-83", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-105", 0 ],
					"source" : [ "obj-84", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-43", 5 ],
					"source" : [ "obj-86", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-43", 7 ],
					"source" : [ "obj-87", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-49", 21 ],
					"source" : [ "obj-90", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-49", 18 ],
					"source" : [ "obj-92", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-49", 19 ],
					"source" : [ "obj-94", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-105", 0 ],
					"source" : [ "obj-96", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-105", 0 ],
					"source" : [ "obj-98", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-105", 0 ],
					"source" : [ "obj-99", 0 ]
				}

			}
 ],
		"parameters" : 		{
			"obj-8" : [ "live.gain~", "live.gain~", 0 ],
			"parameterbanks" : 			{

			}

		}
,
		"dependency_cache" : [ 			{
				"name" : "jg.pvoice.mod1~.maxpat",
				"bootpath" : "~/Documents/Max 8/Library/jg.MaxSynthLib-master/synths/voices",
				"patcherrelativepath" : "../../../../../../Documents/Max 8/Library/jg.MaxSynthLib-master/synths/voices",
				"type" : "JSON",
				"implicit" : 1
			}
, 			{
				"name" : "jg.voice.mod1~.maxpat",
				"bootpath" : "~/Documents/Max 8/Library/jg.MaxSynthLib-master/synths/voices",
				"patcherrelativepath" : "../../../../../../Documents/Max 8/Library/jg.MaxSynthLib-master/synths/voices",
				"type" : "JSON",
				"implicit" : 1
			}
, 			{
				"name" : "jg.voice.mod0~.maxpat",
				"bootpath" : "~/Documents/Max 8/Library/jg.MaxSynthLib-master/synths/voices",
				"patcherrelativepath" : "../../../../../../Documents/Max 8/Library/jg.MaxSynthLib-master/synths/voices",
				"type" : "JSON",
				"implicit" : 1
			}
, 			{
				"name" : "jg.dco~.maxpat",
				"bootpath" : "~/Documents/Max 8/Library/jg.MaxSynthLib-master/modules",
				"patcherrelativepath" : "../../../../../../Documents/Max 8/Library/jg.MaxSynthLib-master/modules",
				"type" : "JSON",
				"implicit" : 1
			}
, 			{
				"name" : "jg.dcsvf~.maxpat",
				"bootpath" : "~/Documents/Max 8/Library/jg.MaxSynthLib-master/modules",
				"patcherrelativepath" : "../../../../../../Documents/Max 8/Library/jg.MaxSynthLib-master/modules",
				"type" : "JSON",
				"implicit" : 1
			}
, 			{
				"name" : "jg.mtoa.maxpat",
				"bootpath" : "~/Documents/Max 8/Library/jg.MaxSynthLib-master/util",
				"patcherrelativepath" : "../../../../../../Documents/Max 8/Library/jg.MaxSynthLib-master/util",
				"type" : "JSON",
				"implicit" : 1
			}
, 			{
				"name" : "jg.mix~.maxpat",
				"bootpath" : "~/Documents/Max 8/Library/jg.MaxSynthLib-master/util",
				"patcherrelativepath" : "../../../../../../Documents/Max 8/Library/jg.MaxSynthLib-master/util",
				"type" : "JSON",
				"implicit" : 1
			}
, 			{
				"name" : "jg.line~.maxpat",
				"bootpath" : "~/Documents/Max 8/Library/jg.MaxSynthLib-master/util",
				"patcherrelativepath" : "../../../../../../Documents/Max 8/Library/jg.MaxSynthLib-master/util",
				"type" : "JSON",
				"implicit" : 1
			}
, 			{
				"name" : "jg.adsrlfo~.maxpat",
				"bootpath" : "~/Documents/Max 8/Library/jg.MaxSynthLib-master/modules",
				"patcherrelativepath" : "../../../../../../Documents/Max 8/Library/jg.MaxSynthLib-master/modules",
				"type" : "JSON",
				"implicit" : 1
			}
, 			{
				"name" : "jg.egenlfo.maxpat",
				"bootpath" : "~/Documents/Max 8/Library/jg.MaxSynthLib-master/modules",
				"patcherrelativepath" : "../../../../../../Documents/Max 8/Library/jg.MaxSynthLib-master/modules",
				"type" : "JSON",
				"implicit" : 1
			}
, 			{
				"name" : "jg.mtof.maxpat",
				"bootpath" : "~/Documents/Max 8/Library/jg.MaxSynthLib-master/util",
				"patcherrelativepath" : "../../../../../../Documents/Max 8/Library/jg.MaxSynthLib-master/util",
				"type" : "JSON",
				"implicit" : 1
			}
, 			{
				"name" : "jg.adsrs~.maxpat",
				"bootpath" : "~/Documents/Max 8/Library/jg.MaxSynthLib-master/modules",
				"patcherrelativepath" : "../../../../../../Documents/Max 8/Library/jg.MaxSynthLib-master/modules",
				"type" : "JSON",
				"implicit" : 1
			}
, 			{
				"name" : "jg.ms.js",
				"bootpath" : "~/Documents/Max 8/Library/jg.MaxSynthLib-master/util",
				"patcherrelativepath" : "../../../../../../Documents/Max 8/Library/jg.MaxSynthLib-master/util",
				"type" : "TEXT",
				"implicit" : 1
			}
, 			{
				"name" : "timbremapctrl.js",
				"bootpath" : "~/Documents/Max 8/Library/jg.MaxSynthLib-master/timbremap",
				"patcherrelativepath" : "../../../../../../Documents/Max 8/Library/jg.MaxSynthLib-master/timbremap",
				"type" : "TEXT",
				"implicit" : 1
			}
, 			{
				"name" : "jg.relpath.maxpat",
				"bootpath" : "~/Documents/Max 8/Library/jg.MaxSynthLib-master/util",
				"patcherrelativepath" : "../../../../../../Documents/Max 8/Library/jg.MaxSynthLib-master/util",
				"type" : "JSON",
				"implicit" : 1
			}
, 			{
				"name" : "generator_rand.js",
				"bootpath" : "~/Documents/Max 8/Library/jg.MaxSynthLib-master/timbremap",
				"patcherrelativepath" : "../../../../../../Documents/Max 8/Library/jg.MaxSynthLib-master/timbremap",
				"type" : "TEXT",
				"implicit" : 1
			}
, 			{
				"name" : "blocks.pad.mxo",
				"type" : "iLaX"
			}
, 			{
				"name" : "jg.mfe~.mxo",
				"type" : "iLaX"
			}
, 			{
				"name" : "jg.modlin~.mxo",
				"type" : "iLaX"
			}
, 			{
				"name" : "jg.egen~.mxo",
				"type" : "iLaX"
			}
, 			{
				"name" : "blocks.multitouch.mxo",
				"type" : "iLaX"
			}
, 			{
				"name" : "jg.timbremap.mxo",
				"type" : "iLaX"
			}
 ],
		"autosave" : 0,
		"boxgroups" : [ 			{
				"boxes" : [ "obj-80", "obj-53", "obj-51", "obj-73", "obj-54", "obj-52", "obj-35", "obj-56", "obj-68", "obj-71", "obj-65", "obj-63", "obj-55", "obj-72", "obj-34", "obj-81" ]
			}
, 			{
				"boxes" : [ "obj-89", "obj-88", "obj-74", "obj-90", "obj-100", "obj-95", "obj-91", "obj-93", "obj-92", "obj-94", "obj-181", "obj-165" ]
			}
, 			{
				"boxes" : [ "obj-18", "obj-59", "obj-60", "obj-61", "obj-69", "obj-70", "obj-23", "obj-33", "obj-66", "obj-50", "obj-67", "obj-36", "obj-41" ]
			}
, 			{
				"boxes" : [ "obj-25", "obj-2", "obj-1", "obj-6", "obj-15", "obj-21", "obj-22", "obj-17" ]
			}
, 			{
				"boxes" : [ "obj-16", "obj-7", "obj-10", "obj-13", "obj-14", "obj-5", "obj-9", "obj-101", "obj-11", "obj-3" ]
			}
, 			{
				"boxes" : [ "obj-130", "obj-129", "obj-39", "obj-128", "obj-32" ]
			}
 ]
	}

}
