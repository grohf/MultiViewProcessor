/*
 * module.h
 *
 *  Created on: Aug 22, 2012
 *      Author: avo
 */

#ifndef MODULE_H_
#define MODULE_H_

class Module {
public:
	virtual ~Module() {}

	virtual void init() = 0;
	virtual void execute() = 0;
};

#endif /* MODULE_H_ */
