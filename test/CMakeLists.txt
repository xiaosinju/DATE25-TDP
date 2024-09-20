cmake_minimum_required(VERSION 3.0.2)

project(test)

install(
    DIRECTORY iccad2015.ot DESTINATION ${PROJECT_NAME}
    )

install(
    DIRECTORY iccad2015.pin2pin DESTINATION ${PROJECT_NAME}
    )
