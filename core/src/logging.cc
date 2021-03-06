/*
 *  MoMEMta: a modular implementation of the Matrix Element Method
 *  Copyright (C) 2016  Universite catholique de Louvain (UCL), Belgium
 *
 *  This program is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation, either version 3 of the License, or
 *  (at your option) any later version.
 *
 *  This program is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

#include <momemta/Logging.h>

#include <memory>
#include <unistd.h>

#include <logger/stdout_sink.h>

namespace logger {
static logger_ptr init_logger() {
    bool in_terminal = isatty(fileno(stdout)) == 1;

    auto sink = sinks::stdout_sink_st::instance();

    auto l = std::make_shared<logger>(sink);
    l->flush_on(logging::level::trace);

    if (in_terminal) {
        l->set_formatter(std::make_shared<ansi_color_full_formatter>());
    }

    return l;
}

logger_ptr get() {
    static logger_ptr s_logger = init_logger();
    return s_logger;
}

}

namespace logging {
    void set_level(::logging::level::level_enum lvl) {
        ::logger::get()->set_level(lvl);
    }
}